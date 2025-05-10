import json
import random
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

TRANSLATED_FILE = "annotations_translated.json"

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

def print_eval_singular(language, y_answer, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_answer, y_pred)

    prec_no, prec_yes = precision
    rec_no, rec_yes = recall
    f1_no, f1_yes = f1

    print(f"{language.capitalize()} language:")
    print("Overall: acc: {:.3f}, f1: {:.3f}".format(accuracy_score(y_answer, y_pred), (f1_yes + f1_no) / 2))
    print("    Yes: prec: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(prec_yes, rec_yes, f1_yes))
    print("     No: prec: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(prec_no, rec_no, f1_no))
    print()

def print_eval(data, y_pred):
    """
    Function to print overall and per language results
    Args:
    data: raw data in list of dict format, not only the label
    y_pred: the predicted label
    """
    if len(data) != len(y_pred):
        print("data and y_pred have to be the same length and in the same order")
        return

    eval_data = [(datum['language'], datum['label'], pred) for datum, pred in zip(data, y_pred)]
    print_eval_singular("all", [datum[1] for datum in eval_data], [datum[2] for datum in eval_data])

    # Split into per languages
    data_per_languages = dict()

    for datum in eval_data:
        if datum[0] not in data_per_languages:
            data_per_languages[datum[0]] = ([datum[1]], [datum[2]])
            continue
        data_per_languages[datum[0]][0].append(datum[1])
        data_per_languages[datum[0]][1].append(datum[2])

    # Print eval for each languages
    for k, v in data_per_languages.items():
        print_eval_singular(k, v[0], v[1])

def load_data(file_path, train_split, val_split=None):
    # Set val_split if not defined
    if val_split is None:
        val_split = 1 - train_split

    with open(file_path, "r") as f:
        data = json.load(f)

    data = [datum for datum in data if datum['label'] != -1] # TODO: Temporary fix
    data = [datum for datum in data if 'translated' in datum]

    # Shuffle data
    random.seed(42)
    random.shuffle(data)

    # Split data into languages
    data_per_languages = dict()

    for datum in data:
        if datum['language'] not in data_per_languages:
            data_per_languages[datum['language']] = [datum]
            continue
        data_per_languages[datum['language']].append(datum)

    # Fill train data split
    train_data = []
    val_data = []
    test_data = []

    for data_per_language in data_per_languages.values():
        train_count = int(len(data_per_language) * train_split)
        val_count = int(len(data_per_language) * val_split)

        train_data.extend(data_per_language[:train_count])
        val_data.extend(data_per_language[train_count:train_count + val_count])
        test_data.extend(data_per_language[train_count + val_count:])

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data

def generate_prompt(data, context=None):
    # TODO: Handle context
    return [f"The next phrase comes from a {datum['language'].capitalize()} speaker: {datum['translated']}" for datum in data]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    # Load data
    train_data, val_data, test_data = load_data(TRANSLATED_FILE, 0.5)

    # Start fine tuning
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = BinaryClassifier(AutoModel.from_pretrained("distilroberta-base"))
    model.to(device)

    # Prepare dataset
    X_train = generate_prompt(train_data, None)
    X_val = generate_prompt(val_data, None)

    y_train = [datum['label'] for datum in train_data]

    # Tokenize
    X_val_tensor = tokenizer(X_val, padding=True, truncation=True, return_tensors="pt")

    val_loader = TensorDataset(X_val_tensor['input_ids'], X_val_tensor['attention_mask'])
    val_loader = DataLoader(val_loader, batch_size=256, shuffle=False)

    X_train_tensor = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    y_train_tensor = torch.tensor([float(y) for y in y_train])

    train_loader = TensorDataset(X_train_tensor['input_ids'], X_train_tensor['attention_mask'], y_train_tensor)
    train_loader = DataLoader(train_loader, batch_size=256, shuffle=False)

    # Finetune
    losses = finetune_binary_classifier(model, train_loader, 1, 1e-5, 1e-3)
    preds = evaluate_binary_classifier(model, val_loader)

    print(f"Results for {model.__class__.__name__}")
    plt.plot(losses)
    plt.title(f"{model.__class__.__name__} loss curve")
    plt.show()
    print_eval(val_data, preds)
