import json
import random
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

TRANSLATED_FILE = "annotations_translated.json"

def print_eval(y_pred, y_answer):
    precision, recall, f1, _ = precision_recall_fscore_support(y_answer, y_pred)

    prec_no, prec_yes = precision
    rec_no, rec_yes = recall
    f1_no, f1_yes = f1

    print("Overall: acc: {:.3f}, f1: {:.3f}".format(accuracy_score(y_answer, y_pred), (f1_yes + f1_no) / 2))
    print("    Yes: prec: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(prec_yes, rec_yes, f1_yes))
    print("     No: prec: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(prec_no, rec_no, f1_no))

class BinaryClassifier(torch.nn.Module):
    def __init__(self, base_model):
        super(BinaryClassifier, self).__init__()
        self.base = base_model
        self.classifier = torch.nn.Linear(self.base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_repr = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token_repr)

def finetune_binary_classifier(model, train_loader, num_epochs, lr, weight_decay):
    """
    Function to fine-tune a model to do a binary classification task
    Args:
    model: instance of BinaryClassifier
    train_loader: Dataloader for the dataset
    num_epochs: Number of epochs for training
    lr: Learning rate
    weight_decay: Weight decay

    Returns:
    batch_losses: List of losses for each mini-batch
    """
    device = next(model.parameters()).device
    model.train()
    batch_losses = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for _ in range(num_epochs):
        for (input_ids, attention_mask, labels) in tqdm(train_loader):
            optimizer.zero_grad()

            # Trim input
            seq_len = attention_mask.sum(dim=1).max()
            input_ids = input_ids[:, -seq_len:]
            attention_mask = attention_mask[:, -seq_len:]

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                loss = loss_func(logits.squeeze(1), labels.to(device))

            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

    return batch_losses

def evaluate_binary_classifier(model, data_loader):
    """
    Function to implement BinaryClassifier inference
    Args:
    model: instance of BinaryClassifier
    data_loader: Dataloader for the dataset

    Returns:
    preds
    """
    device = next(model.parameters()).device
    model.eval()
    preds = []

    for (input_ids, attention_mask) in tqdm(data_loader):
        # Trim input
        seq_len = attention_mask.sum(dim=1).max()
        input_ids = input_ids[:, -seq_len:]
        attention_mask = attention_mask[:, -seq_len:]

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            pred = (torch.sigmoid(logits).squeeze(1) >= 0.5).tolist()

        preds.extend(pred)
        del logits

    return preds

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    with open(TRANSLATED_FILE, "r") as f:
        data = json.load(f)

    data = [datum for datum in data if datum['label'] != -1]
    data = [datum for datum in data if 'translated' in datum]

    random.seed(42)
    random.shuffle(data)

    X = [f"The next phrase comes from a {datum['language'].capitalize()} speaker: {datum['text']}" for datum in data]
    y = [datum['label'] for datum in data]

    X_train = X[:len(X) // 2]
    y_train = y[:len(y) // 2]

    X_val = X[len(X) // 2:]
    y_val = y[len(y) // 2:]

    torch.manual_seed(42)

    # Start fine tuning
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    model = BinaryClassifier(AutoModel.from_pretrained("distilroberta-base"))
    model.to(device)

    X_val_roberta = tokenizer(X_val, padding=True, truncation=True)

    val_loader_roberta = TensorDataset(torch.tensor(X_val_roberta['input_ids']), torch.tensor(X_val_roberta['attention_mask']))
    val_loader_roberta = DataLoader(val_loader_roberta, batch_size=256, shuffle=False)

    X_train_roberta = tokenizer(X_train, padding=True, truncation=True)
    y_train_roberta = [float(y) for y in y_train]

    train_loader_roberta = TensorDataset(torch.tensor(X_train_roberta['input_ids']), torch.tensor(X_train_roberta['attention_mask']), torch.tensor(y_train_roberta))
    train_loader_roberta = DataLoader(train_loader_roberta, batch_size=256, shuffle=False)

    losses = finetune_binary_classifier(model, train_loader_roberta, 1, 1e-5, 1e-3)

    preds = evaluate_binary_classifier(model, val_loader_roberta)
    print()
    print("Checkpoint 1.4:")
    plt.plot(losses)
    plt.title("distilroberta loss curve")
    plt.show()
    print_eval(preds, y_val)
    print()
