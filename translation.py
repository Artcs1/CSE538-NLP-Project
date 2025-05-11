############################################
# CSE538 Natural Language Processing
# Team name: Decepticons
# Team members:
# - Enamul Hassan
# - Jeffri Murrugarra
# - Kevin Dharmawan
# 
# Translates the texts in the combined data
# Implements Part 1 (tokenization), Part 3 (generative)
# Note that this is not the main part of our project
# The main part is in finetune.py
############################################

import config
import json
import torch

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def translate(model, tokenizer, texts, target_lang):
    """
    Translate texts to target language.
    
    Args:
        model: Google MadLad model.
        tokenizer: Google MadLad tokenizer.
        texts: List of text to be translated.
        target_lang: 2 character target language code.
    
    Returns:
        list of string: Translated version of texts.
    """

    device = next(model.parameters()).device

    inputs = tokenizer(
        [f"<2{target_lang}> {text}" for text in texts],
        return_tensors="pt",
        max_length=config.TRANSLATION_MAX_LENGTH,
        padding=True,
        truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Default max length is 21, thus we need to change it
    # Translation output multiplier is used in case the output is longer than the input
    input_length = inputs['input_ids'].shape[1]
    max_length = min(int(input_length * config.TRANSLATION_OUTPUT_LENGTH_MULTIPLIER), config.TRANSLATION_MAX_LENGTH)

    # PART 3: GENERATIVE
    outputs = model.generate(**inputs, max_length=max_length)
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return translations

def translate_data(model, tokenizer, data, batch_size):
    """
    Translate texts in data to English. Translated text will be added to dict
    in data in-place.
    
    Args:
        model: Google MadLad model.
        tokenizer: Google MadLad tokenizer.
        data: Data in the form of list of dict.
        batch_size: Translation batch size.
    """
    N = len(data)

    for i in tqdm(range(N // batch_size + (1 if N % batch_size else 0))):
        texts = [x['text'] for x in data[i * batch_size:(i + 1) * batch_size]]
        translated_texts = translate(model, tokenizer, texts, "en")

        for j in range(len(translated_texts)):
            data[(i * batch_size) + j]['translated'] = translated_texts[j]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-3b-mt", torch_dtype=torch.float16)
    model.to(device)
    # PART 1: TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")

    # Read combined data source
    with open(config.PREPROCESS_OUTPUT_PATH, "r") as f:
        data = json.load(f)

    # Do translation
    translate_data(model, tokenizer, data, config.TRANSLATION_BATCH_SIZE)

    # Export the combined data source with the translation
    with open(config.TRANSLATION_OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=4)
