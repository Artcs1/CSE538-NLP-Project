import json
import torch

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_LENGTH = 1024
OUTPUT_LENGTH_MULTIPLIER = 1.5
BATCH_SIZE = 64
INPUT_FILE = "annotations.json"
OUTPUT_FILE = "annotations_translated.json"

def translate(model, tokenizer, texts, target_lang):
    device = next(model.parameters()).device

    inputs = tokenizer(
        [f"<2{target_lang}> {text}" for text in texts],
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Default max length is 21, thus we need to change it
    input_length = inputs['input_ids'].shape[1]
    max_length = min(int(input_length * OUTPUT_LENGTH_MULTIPLIER), MAX_LENGTH)

    outputs = model.generate(**inputs, max_length=max_length)
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return translations

def translate_data(model, tokenizer, data, batch_size):
    N = len(data)

    for i in tqdm(range(N // batch_size)):
        texts = [x['text'] for x in data[i * batch_size:(i + 1) * batch_size]]
        translated_texts = translate(model, tokenizer, texts, "en")

        for j in range(batch_size):
            data[(i * batch_size) + j]['translated'] = translated_texts[j]

if __name__ == "__main__":
    device = 'cuda'

    model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-3b-mt", torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    translate_data(model, tokenizer, data, BATCH_SIZE)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=4)
