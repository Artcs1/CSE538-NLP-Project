import os
import json
import random
import torch
import matplotlib.pyplot as plt

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

def grid_search_tuning(model, train_dataset, val_dataset, num_epoch, learning_rates, l2_penalties):
    """
    Grid Search for Hyperparameter Tuning
    
    Args:
        model: The model to be trained (passed as parameter).
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        learning_rates: List of learning rates to try.
        l2_penalties: List of L2 weight decay values to try.
    
    Returns:
        model_accuracies: A 2D list of accuracy values for each hyperparameter combination.
        best_lr: The best learning rate found.
        best_l2_penalty: The best L2 penalty (weight decay) found.
    """
    # Initialize accuracy table
    model_accuracies = [[0.0 for _ in range(len(l2_penalties))] for _ in range(len(learning_rates))]
    highest_acc = (0, 0, 0)  # (best_acc, best_lr_index, best_l2_index)
    data_collator = DataCollatorWithPadding(tokenizer)

    for i, cur_lr in enumerate(learning_rates):
        for j, cur_l2 in enumerate(l2_penalties):
            print(f"Training with LR={cur_lr}, L2 Penalty (Weight Decay)={cur_l2}")
            
            # Set Training Arguments
            training_args = TrainingArguments(
                output_dir="./results",
                learning_rate=cur_lr,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=num_epoch,
                weight_decay=cur_l2
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator
            )

            # Train and evaluate
            trainer.train()
            eval_results = trainer.evaluate()
            accuracy = eval_results.get("eval_accuracy", 0.0)  # Use accuracy from evaluation
            
            model_accuracies[i][j] = accuracy
            print(f"Accuracy: {accuracy:.4f}\n")
            
            # Update highest accuracy
            if accuracy > highest_acc[0]:
                highest_acc = (accuracy, i, j)
    
    # Extract best hyperparameters
    best_lr = learning_rates[highest_acc[1]]
    best_l2_penalty = l2_penalties[highest_acc[2]]

    # Print Results
    print("\nGrid Search Results (Dev Accuracy %):")
    print("LR\\L2\t\t", "\t".join([f"{l2:.0e}" for l2 in l2_penalties]))

    for i, lr in enumerate(learning_rates):
        row = [f"{lr:-4.1e}\t"]
        for j, l2 in enumerate(l2_penalties):
            accuracy = model_accuracies[i][j] * 100
            row.append(f"{accuracy:.2f}")
        print("\t".join(row))

    print(f"\nBest Hyperparameters: LR={best_lr}, L2 Penalty={best_l2_penalty}\n")
    return model_accuracies, best_lr, best_l2_penalty

def generate_prompt(data, translate=False, context=None):
    # TODO: Handle context
    if not translate:

        if context == 'long':
            with open('cultural-context/culture_context_long_orig.jsonl', "r") as f:
                context = json.load(f)
            return [f"Consider that {context[datum['language'].lower()]} and the speaker said: {datum['text']}" for datum in data]
        elif context == 'short':
            with open('cultural-context/culture_context_short_orig.jsonl', "r") as f:
                context = json.load(f)
            return [f"Consider that {context[datum['language'].lower()]} and the speaker said: {datum['text']}" for datum in data]
        elif context == 'graph':
            with open('cultural-context/culture_context_short_orig.jsonl', "r") as f:
                context = json.load(f)
            return [f"Consider that {context[datum['language'].lower()]} and the speaker said: {datum['text']}" for datum in data]
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
    else:
        return [f"The next phrase comes from a {datum['language'].capitalize()} speaker: {datum['translated']}" for datum in data]

    return [f"The next phrase comes from a {datum['language'].capitalize()} speaker: {datum['translated']}" for datum in data]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    os.environ["WANDB_DISABLED"] = "true"

    # Load data
    train_data, val_data, test_data = load_data(TRANSLATED_FILE, 0.5)

    # Start fine tuning
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = BinaryClassifier(AutoModel.from_pretrained("distilroberta-base"))
    model.to(device)

    # Prepare dataset
    X_train = generate_prompt(train_data, True, None)
    X_val = generate_prompt(val_data, True, None)

    y_train = [datum['label'] for datum in train_data]
    y_val = [datum['label'] for datum in val_data]


    # Tokenize
    X_val_tensor = tokenizer(X_val, padding=True, truncation=True, return_tensors="pt")
    y_val_tensor = torch.tensor([float(y) for y in y_val])

    val_loader = TensorDataset(X_val_tensor['input_ids'], X_val_tensor['attention_mask'])
    val_loader = DataLoader(val_loader, batch_size=256, shuffle=False)

    X_train_tensor = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    y_train_tensor = torch.tensor([float(y) for y in y_train])

    train_loader = TensorDataset(X_train_tensor['input_ids'], X_train_tensor['attention_mask'], y_train_tensor)
    train_loader = DataLoader(train_loader, batch_size=256, shuffle=False)

#    search_train_data = [
#        {
#            "input_ids": X_train_tensor['input_ids'][i],
#            "attention_mask": X_train_tensor['attention_mask'][i],
#            "labels": y_train_tensor[i]
#        }
#        for i in range(len(y_train_tensor))
#    ]
#
#    search_val_data = [
#        {
#            "input_ids": X_val_tensor['input_ids'][i],
#            "attention_mask": X_val_tensor['attention_mask'][i],
#            "labels": y_val_tensor[i]
#        }
#        for i in range(len(y_val_tensor))
#    ]
#    learning_rates = [1e-5, 3e-5, 5e-5]
#    l2_penalties = [1e-5, 1e-3, 1e-1]
#    
#    # Run Grid Search
#    model_accuracies, best_lr, best_l2_penalty = grid_search_tuning(
#        model, search_train_data, search_val_data, num_epoch=1, learning_rates=learning_rates, l2_penalties=l2_penalties
#    )
#    
#    print(f"Best Hyperparameters: LR={best_lr}, L2 Penalty={best_l2_penalty}\n")
#
    # Finetune with Best Hyperparameters

    best_lr = 1e-5
    best_l2_penalty = 1e-3

    losses = finetune_binary_classifier(model, train_loader, 1, best_lr, best_l2_penalty)
    preds = evaluate_binary_classifier(model, val_loader)

    print(f"Results for {model.__class__.__name__}")
    plt.plot(losses)
    plt.title(f"{model.__class__.__name__} loss curve")
    plt.show()
    print_eval(val_data, preds)
