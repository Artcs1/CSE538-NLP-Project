import json
import torch
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TRANSLATED_FILE = "annotations_translated.json"
MODEL_NAME     = "distilroberta-base"
NUM_EPOCHS     = 3
LR             = 1e-5
WEIGHT_DECAY   = 1e-3
BATCH_SIZE     = 16
TEST_SIZE      = 0.10   # 10% for test
VAL_SIZE       = 0.20   # 20% of the remaining train+val for validation

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def print_eval(y_true, y_pred):
    """
    Print overall accuracy & macro-F1, then per-label precision/recall/F1 for
    label 0 (No) and label 1 (Yes).
    """
    # overall
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    # f1[0] → No, f1[1] → Yes
    macro_f1 = (f1[0] + f1[1]) / 2

    print(f"Overall: acc: {acc:.3f}, f1: {macro_f1:.3f}")
    print(f"    Yes: prec: {prec[1]:.3f}, rec: {rec[1]:.3f}, f1: {f1[1]:.3f}")
    print(f"     No: prec: {prec[0]:.3f}, rec: {rec[0]:.3f}, f1: {f1[0]:.3f}")
    print()

def evaluate_binary_classifier(model, data_loader, device):
    """
    Function to implement BinaryClassifier inference
    Args:
    model: instance of BinaryClassifier
    data_loader: Dataloader for the dataset
    device: Device to run the model on (CPU or GPU)

    Returns:
    preds
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in tqdm(data_loader, desc="Evaluating"):
            # trim
            seq_len = attention_mask.sum(dim=1).max()
            input_ids     = input_ids[:, -seq_len:]
            attention_mask= attention_mask[:, -seq_len:]

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                pred = (torch.sigmoid(logits).squeeze(1) >= 0.5).tolist()

            preds.extend(pred)
            del logits
    return preds

class BinaryClassifier(torch.nn.Module):
    """
    Simple binary classifier on top of a transformer encoder.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base       = base_model
        self.classifier = torch.nn.Linear(self.base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr= outputs.last_hidden_state[:, 0]
        return self.classifier(cls_repr)

def finetune_binary_classifier(model, train_loader, val_loader, num_epochs, lr, weight_decay, device):
    """
    Function to fine-tune a model to do a binary classification task
    Args:
    model: instance of BinaryClassifier
    train_loader: Dataloader for the dataset
    val_loader: Dataloader for the validation set
    num_epochs: Number of epochs for training
    lr: Learning rate
    weight_decay: Weight decay
    device: Device to run the model on (CPU or GPU)

    Returns:
    batch_losses: List of losses for each mini-batch
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = torch.nn.BCEWithLogitsLoss()
    batch_losses = []

    for epoch in range(num_epochs):
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            # Trim input
            seq_len = attention_mask.sum(dim=1).max()
            input_ids      = input_ids[:, -seq_len:]
            attention_mask = attention_mask[:, -seq_len:]

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                loss = loss_fn(logits.squeeze(1), labels.to(device).float())

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # ——— validation pass ———
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                seq_len = attention_mask.sum(dim=1).max()
                input_ids      = input_ids[:, -seq_len:].to(device)
                attention_mask = attention_mask[:, -seq_len:].to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(1)
                    probs  = torch.sigmoid(logits)

                preds = (probs >= 0.5).long().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        print(f"\n--- Validation after epoch {epoch+1} ---")
        print_eval(all_preds, all_labels)
        model.train()

    return batch_losses

def tune_hyperparameters(train_loader, val_loader, device):
    """
    Loop over a small grid, return best (model, config).
    """
    param_grid = [
        {"lr": 1e-5, "weight_decay": 1e-3, "num_epochs": 3, "batch_size": 16},
        {"lr": 5e-6, "weight_decay": 0,    "num_epochs": 3, "batch_size": 16},
        {"lr": 1e-4, "weight_decay": 1e-4, "num_epochs": 3, "batch_size": 32},
        # …add more combos or generate via itertools.product…
    ]

    best_f1   = 0.0
    best_cfg  = None
    best_model= None

    for cfg in param_grid:
        # rebuild loaders if batch_size changes
        if train_loader.batch_size != cfg["batch_size"]:
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=cfg["batch_size"],
                shuffle=True
            )
            val_loader   = DataLoader(
                val_loader.dataset,
                batch_size=cfg["batch_size"]
            )

        model = BinaryClassifier(AutoModel.from_pretrained(MODEL_NAME))
        finetune_binary_classifier(
            model,
            train_loader,
            val_loader,
            cfg["num_epochs"],
            cfg["lr"],
            cfg["weight_decay"],
            device
        )

        preds = evaluate_binary_classifier(model, val_loader, device)
        _, _, f1_scores, _ = precision_recall_fscore_support(
            [t for _,_,t in val_loader.dataset],
            preds, zero_division=0
        )
        macro_f1 = (f1_scores[0] + f1_scores[1]) / 2

        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_cfg   = cfg
            best_model = model

    print("Best config:", best_cfg, "Val macro-F1:", best_f1)
    return best_model, best_cfg



def make_dataset(texts, labels, tokenizer):
    """
    Tokenize and wrap into a TensorDataset.
    """
    enc = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    return TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(labels, dtype=torch.long)
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load & filter
    with open(TRANSLATED_FILE, "r") as f:
        data = json.load(f)
    data = [d for d in data if d.get("label", -1) != -1 and "translated" in d]

    # 2) Prepare labels & languages
    y     = [d["label"] for d in data]
    langs = [d["language"] for d in data]

    # 3) Build X for baseline vs. context
    X_baseline = [d["translated"] for d in data]
    X_context  = [
        f"The next phrase comes from a {lang.capitalize()} speaker: {d['translated']}"
        for d, lang in zip(data, langs)
    ]

    # 4) Stratified split: train+val / test
    X_tv_base, X_test_base, y_tv_base, y_test_base, lang_tv, lang_test = train_test_split(
        X_baseline, y, langs,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=langs
    )
    X_tv_ctx, X_test_ctx, y_tv_ctx, y_test_ctx, _, _  = train_test_split(
        X_context,  y, langs,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=langs
    )

    # 5) Stratified split: train / val (on the remaining 1 - TEST_SIZE)
    X_train_base, X_val_base, y_train_base, y_val_base, _, lang_val = train_test_split(
        X_tv_base, y_tv_base, lang_tv,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=lang_tv
    )
    X_train_ctx, X_val_ctx, y_train_ctx, y_val_ctx, _, _ = train_test_split(
        X_tv_ctx, y_tv_ctx, lang_tv,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=lang_tv
    )

    # 6) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 7) Build datasets
    train_ds_base = make_dataset(X_train_base, y_train_base, tokenizer)
    val_ds_base   = make_dataset(X_val_base,   y_val_base,   tokenizer)
    test_ds_base  = make_dataset(X_test_base,  y_test_base,  tokenizer)

    train_ds_ctx  = make_dataset(X_train_ctx,  y_train_ctx,  tokenizer)
    val_ds_ctx    = make_dataset(X_val_ctx,    y_val_ctx,    tokenizer)
    test_ds_ctx   = make_dataset(X_test_ctx,   y_test_ctx,   tokenizer)

    # 8) Build dataloaders
    train_loader_base = DataLoader(train_ds_base, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_base   = DataLoader(val_ds_base,   batch_size=BATCH_SIZE)
    test_loader_base  = DataLoader(test_ds_base,  batch_size=BATCH_SIZE)

    train_loader_ctx  = DataLoader(train_ds_ctx,  batch_size=BATCH_SIZE, shuffle=True)
    val_loader_ctx    = DataLoader(val_ds_ctx,    batch_size=BATCH_SIZE)
    test_loader_ctx   = DataLoader(test_ds_ctx,   batch_size=BATCH_SIZE)

    # -----------------------------------------------------------------------------
    # 9) Baseline (no context)
    # -----------------------------------------------------------------------------
    # ---- hyperparameter tuning for baseline ----
    model_base, best_cfg = tune_hyperparameters(
        train_loader_base,
        val_loader_base,
        device
    )
    print("Best baseline config:", best_cfg)

    # Final test evaluation baseline
    print("=== BASELINE — FINAL TEST ===")
    preds_test_base = evaluate_binary_classifier(model_base, test_loader_base, device)
    print_eval(preds_test_base, y_test_base)

    # -----------------------------------------------------------------------------
    # 10) Contextual (with culture context)
    # -----------------------------------------------------------------------------
    print("\n=== CONTEXTUAL (with culture context) ===\n")
    # Tune on train+val
    model_ctx, best_cfg_ctx = tune_hyperparameters(
        train_loader_ctx,
        val_loader_ctx,
        device
    )
    print("Best contextual config:", best_cfg_ctx)

    # Final test evaluation
    print("=== CONTEXTUAL — FINAL TEST ===")
    preds_test_ctx = evaluate_binary_classifier(model_ctx, test_loader_ctx, device)
    print_eval(preds_test_ctx, y_test_ctx)
