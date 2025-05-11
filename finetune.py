import os
import json
import random
import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding

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
    Function to fine-tune a model to do a binary classification task.

    Args:
        model: instance of BinaryClassifier.
        train_loader: Dataloader for the dataset.
        num_epochs: Number of epochs for training.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        batch_losses: List of losses for each mini-batch.
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

def evaluate_zero_shot(c_model, c_tokenizer, test_data):
    """
    Function to implement BinaryClassifier inference
    Args:
    model: instance of BinaryClassifier
    data_loader: Dataloader for the dataset

    Returns:
    preds
    """

    device = next(model.parameters()).device
    c_model.eval()

    yes_token_id = c_tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token_id = c_tokenizer.encode("no", add_special_tokens=False)[0]
    preds = []

    for passage in tqdm(test_data):

        prompt   = f"{passage}\n Is the previous sentence offensive?\n"
        inputs   = c_tokenizer(prompt, return_tensors="pt", truncation = True).to(device)

        with torch.no_grad():
            outputs = c_model(**inputs)
            logits = outputs.logits
            next_token_logits = logits[0, -1]
        yes_score = next_token_logits[yes_token_id].item()
        no_score  = next_token_logits[no_token_id].item()
        pred = 1 if yes_score > no_score else 0

        preds.append(pred)

    return preds


def evaluate_binary_classifier(model, data_loader):
    """
    Function to implement BinaryClassifier inference.

    Args:
        model: Instance of BinaryClassifier.
        data_loader: Dataloader for the dataset.

    Returns:
        preds: List of predictions. 
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
    """Function to print results of a single language."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_answer, y_pred)

    prec_no, prec_yes = precision
    rec_no, rec_yes = recall
    f1_no, f1_yes = f1

    print(f"{language.capitalize()} language:")
    print("Overall: acc: {:.3f}, f1: {:.3f}".format(accuracy_score(y_answer, y_pred), (f1_yes + f1_no) / 2))
    print("    Yes: prec: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(prec_yes, rec_yes, f1_yes))
    print("     No: prec: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(prec_no, rec_no, f1_no))
    print()
    return accuracy_score(y_answer, y_pred), (f1_yes + f1_no) / 2, prec_yes, rec_yes, f1_yes, prec_no, rec_no, f1_no

def print_eval(data, y_pred):
    """
    Function to print overall and per language results.

    Args:
        data: Raw data in list of dict format, not only the label.
        y_pred: The predicted label.
    """
    results = []

    if len(data) != len(y_pred):
        print("data and y_pred have to be the same length and in the same order")
        return

    eval_data = [(datum['language'], datum['label'], pred) for datum, pred in zip(data, y_pred)]
    acc, f1, prec_yes, rec_yes, f1_yes, prec_no, rec_no, f1_no = print_eval_singular("all", [datum[1] for datum in eval_data], [datum[2] for datum in eval_data])
    results.append({"model": "all", "accuracy": acc, "f1_score": f1, "prec_yes": prec_yes, "rec_yes": rec_yes, "f1_yes": f1_yes, "prec_no": prec_no, "rec_no": rec_no, "f1_no": f1_no})

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
        acc, f1, prec_yes, rec_yes, f1_yes, prec_no, rec_no, f1_no = print_eval_singular(k, v[0], v[1])
        results.append({"model": k, "accuracy": acc, "f1_score": f1, "prec_yes": prec_yes, "rec_yes": rec_yes, "f1_yes": f1_yes, "prec_no": prec_no, "rec_no": rec_no, "f1_no": f1_no})

    return results

def load_data(file_path, train_split, val_split=None):
    """
    Load and split data per language.

    Args:
        file_path: Path to the translated data.
        train_split: Training set split.
        val_split: Validation set split.

    Returns:
        train_split: Training set based on train_split ratio.
        val_split: Validation set based on train_split ratio or defined by val_split.
        test_split: Test set based on train_split and val_split ratio.
    """
    # Set val_split if not defined
    if val_split is None:
        val_split = 1 - train_split

    with open(file_path, "r") as f:
        data = json.load(f)

    data = [datum for datum in data if isinstance(datum['text'], str)] # TODO: Temporary fix
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

def grid_search_tuning(model, train_loader, val_loader, num_epochs, lrs, l2):
    """
    Grid search for hyperparameter tuning.
    
    Args:
        model: The model to be trained (passed as parameter).
        train_loader: The training data loader.
        val_loader: The validation data loader.
        lr: List of learning rates to try.
        l2: List of L2 weight decay values to try.
    
    Returns:
        model_accuracies: A 2D list of accuracy values for each hyperparameter combination.
        best_lr: The best learning rate found.
        best_l2_penalty: The best L2 penalty (weight decay) found.
    """
    model_accuracies = [[0.0]*len(l2) for _ in lrs]
    best_acc, best_lr, best_wd = 0.0, None, None

    for i, lr in enumerate(lrs):
        for j, wd in enumerate(l2):
            print(f"→ Training with lr={lr:.1e}, weight_decay={wd:.1e}")
            # finetune for one trial
            finetune_binary_classifier(model, train_loader, num_epochs, lr, wd)
            # evaluate
            preds = evaluate_binary_classifier(model, val_loader)
            acc = accuracy_score(y_val, preds)
            model_accuracies[i][j] = acc
            print(f"   ↳ val_acc = {acc:.4f}\n")

            if acc > best_acc:
                best_acc, best_lr, best_wd = acc, lr, wd

    # report
    print("Grid Search Results (val accuracy):")
    header = ["    WD→"] + [f"{wd:.0e}" for wd in l2]
    print("\t".join(header))
    for clr, row in zip(lrs, model_accuracies):
        row_str = [f"{clr:.0e}"] + [f"{acc*100:5.1f}%" for acc in row]
        print("\t".join(row_str))

    print(f"\nBest: lr={best_lr:.1e}, weight_decay={best_wd:.1e}  (accuracy={best_acc:.4f})")
    return model_accuracies, best_lr, best_wd


def generate_prompt(data, translate=False, context=None):
    """
    Add context to raw text or translated text.
    
    Args:
        data: Data that will be used to generate prompt.
        translate: Use translated text or original text.
        context: 'long'/'short'/'graph'/'simple'/'none'.
    
    Returns:
        list of string: Final prompt.
    """
    if not translate:
        if context == 'long':
            with open('cultural-context/culture_context_long.jsonl', "r") as f:
                context = json.load(f)
            return [f"Consider that {context[datum['language'].lower()]} and the speaker said: {datum['text']}" for datum in data]
        elif context == 'short':
            with open('cultural-context/culture_context_short.jsonl', "r") as f:
                context = json.load(f)
            return [f"Consider that {context[datum['language'].lower()]} and the speaker said: {datum['text']}" for datum in data]
        elif context == 'graph':
            with open('cultural-context/culture_context_graph1.jsonl', "r") as f:
                context = json.load(f)
            return [f"Consider that {context[datum['language'].lower()]} and the speaker said: {datum['text']}" for datum in data]
        elif context == 'simple':
            return [f"The next phrase comes from a {datum['language'].capitalize()} speaker: {datum['text']}" for datum in data]
        else:
            return [datum['text'] for datum in data]

    if context == 'long':
        with open('cultural-context/culture_context_long.jsonl', "r") as f:
            context = json.load(f)
        return [f"Consider that {context[datum['language'].lower()]} and the next phrase comes from a {datum['language'].capitalize()} speaker: {datum['translated']}" for datum in data]
    elif context == 'short':
        with open('cultural-context/culture_context_short.jsonl', "r") as f:
            context = json.load(f)
        return [f"Consider that {context[datum['language'].lower()]} and the next phrase comes from a {datum['language'].capitalize()} speaker: {datum['translated']}" for datum in data]
    elif context == 'graph':
        with open('cultural-context/culture_context_graph1.jsonl', "r") as f:
            context = json.load(f)
        return [f"Consider that {context[datum['language'].lower()]} and the next phrase comes from a {datum['language'].capitalize()} speaker: {datum['translated']}" for datum in data]
    elif context == 'simple':
        return [f"The next phrase comes from a {datum['language'].capitalize()} speaker: {datum['translated']}" for datum in data]
    else:
        return [f"{datum['translated']}" for datum in data]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Choose the setting.")
    parser.add_argument("--model", type=str, default="distilroberta-base")
    parser.add_argument("--context", type=str, default="None")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--zero_shot", action="store_true")

    args = parser.parse_args()

    print(f'THIS RUN IS WITH model: {args.model}, with context {args.context} and translation {args.translate}')   

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    os.environ["WANDB_DISABLED"] = "true"

    # Load data
    trainval_data, test_data, _ = load_data(TRANSLATED_FILE, 0.5)

    # Start fine tuning
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if "gpt2" in args.model:
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model)
    else:
        model = BinaryClassifier(AutoModel.from_pretrained(args.model))

    model.to(device)

    # Prepare dataset
    X_trainval = generate_prompt(trainval_data, args.translate, args.context)
    X_test = generate_prompt(test_data, args.translate, args.context)

    y_trainval = [datum['label'] for datum in trainval_data]
    y_test = [datum['label'] for datum in test_data]

    # Tokenize
    X_test_tensor = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")
    y_test_tensor = torch.tensor([float(y) for y in y_test])

    test_loader = TensorDataset(X_test_tensor['input_ids'], X_test_tensor['attention_mask'])
    test_loader = DataLoader(test_loader, batch_size=256, shuffle=False)

    X_trainval_tensor = tokenizer(X_trainval, padding=True, truncation=True, return_tensors="pt")
    y_trainval_tensor = torch.tensor([float(y) for y in y_trainval])

    trainval_loader = TensorDataset(X_trainval_tensor['input_ids'], X_trainval_tensor['attention_mask'], y_trainval_tensor)
    trainval_loader = DataLoader(trainval_loader, batch_size=256, shuffle=False)

        
    # Hyperparameter Tuning - Grid Search
    if args.validate:

        midpoint = len(X_trainval) // 2

        X_train = X_trainval[:midpoint]
        X_val = X_trainval[midpoint:]

        y_train = y_trainval[:midpoint]
        y_val = y_trainval[midpoint:]
        
        X_train_tensor = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
        y_train_tensor = torch.tensor([float(y) for y in y_train])

        train_loader = TensorDataset(X_train_tensor['input_ids'], X_train_tensor['attention_mask'], y_train_tensor)
        train_loader = DataLoader(train_loader, batch_size=256, shuffle=False)

        X_val_tensor = tokenizer(X_val, padding=True, truncation=True, return_tensors="pt")
        y_val_tensor = torch.tensor([float(y) for y in y_val])

        val_loader = TensorDataset(X_val_tensor['input_ids'], X_val_tensor['attention_mask'])
        val_loader = DataLoader(val_loader, batch_size=256, shuffle=False)

        learning_rates = [1e-5, 3e-5, 5e-5]
        l2_penalties = [1e-5, 1e-3, 1e-1]
        
        # Run Grid Search
        model_accuracies, best_lr, best_l2_penalty = grid_search_tuning(
            model, train_loader, val_loader, num_epochs=1, lrs=learning_rates, l2=l2_penalties
        )
        
        print(f"Best Hyperparameters: LR={best_lr}, L2 Penalty={best_l2_penalty}\n")

    else:
        best_lr = 1e-5
        best_l2_penalty = 1e-3

    if not args.zero_shot: 

        losses = finetune_binary_classifier(model, trainval_loader, num_epochs=1,
                                            lr=best_lr, weight_decay=best_l2_penalty)
        preds = evaluate_binary_classifier(model, test_loader)

        print(f"Results for {model.__class__.__name__}")
        plt.plot(losses)
        plt.title(f"{model.__class__.__name__} loss curve")
        plt.show()


    else:

        preds = evaluate_zero_shot(model, tokenizer, X_test)
        
    print(f"Results for {model.__class__.__name__}")
    plt.plot(losses)
    plt.title(f"{model.__class__.__name__} loss curve")
    plt.show()

    results = print_eval(test_data, preds)

    dir_name = 'results/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    df = pd.DataFrame(results)
    exp_name = f'model_{args.model}_Translate:{args.translate}_Context:{args.context}_Validate:{args.validate}.csv'
    df.to_csv(dir_name+exp_name, index=False)
