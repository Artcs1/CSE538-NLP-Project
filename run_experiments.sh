#!/bin/bash

python3 finetune.py --model roberta-base --translate --context None
python3 finetune.py --model roberta-base --translate --context simple
python3 finetune.py --model roberta-base --translate --context long
python3 finetune.py --model roberta-base --translate --context short
python3 finetune.py --model roberta-base --translate --context graph

python3 finetune.py --model distilroberta-base --translate --context None
python3 finetune.py --model distilroberta-base --translate --context simple
python3 finetune.py --model distilroberta-base --translate --context long
python3 finetune.py --model distilroberta-base --translate --context short
python3 finetune.py --model distilroberta-base --translate --context graph

python3 finetune.py --model roberta-base --translate --context None --validate
python3 finetune.py --model roberta-base --translate --context simple --validate
python3 finetune.py --model roberta-base --translate --context long --validate
python3 finetune.py --model roberta-base --translate --context short --validate
python3 finetune.py --model roberta-base --translate --context graph --validate

python3 finetune.py --model distilroberta-base --translate --context None --validate
python3 finetune.py --model distilroberta-base --translate --context simple --validate
python3 finetune.py --model distilroberta-base --translate --context long --validate
python3 finetune.py --model distilroberta-base --translate --context short --validate
python3 finetune.py --model distilroberta-base --translate --context graph --validate

python3 finetune.py --model distilbert-base-multilingual-cased --context None
python3 finetune.py --model distilbert-base-multilingual-cased --context simple
python3 finetune.py --model distilbert-base-multilingual-cased --context long
python3 finetune.py --model distilbert-base-multilingual-cased --context short
python3 finetune.py --model distilbert-base-multilingual-cased --context graph

python3 finetune.py --model bert-base-multilingual-cased --context None
python3 finetune.py --model bert-base-multilingual-cased --context simple 
python3 finetune.py --model bert-base-multilingual-cased --context long
python3 finetune.py --model bert-base-multilingual-cased --context short
python3 finetune.py --model bert-base-multilingual-cased --context graph

python3 finetune.py --model distilbert-base-multilingual-cased --context None --validate
python3 finetune.py --model distilbert-base-multilingual-cased --context simple --validate
python3 finetune.py --model distilbert-base-multilingual-cased --context long --validate
python3 finetune.py --model distilbert-base-multilingual-cased --context short --validate
python3 finetune.py --model distilbert-base-multilingual-cased --context graph --validate

python3 finetune.py --model bert-base-multilingual-cased --context None --validate
python3 finetune.py --model bert-base-multilingual-cased --context simple --validate
python3 finetune.py --model bert-base-multilingual-cased --context long --validate
python3 finetune.py --model bert-base-multilingual-cased --context short --validate
python3 finetune.py --model bert-base-multilingual-cased --context graph --validate


python3 finetune.py --model distilbert-base-multilingual-uncased --context None
python3 finetune.py --model distilbert-base-multilingual-uncased --context simple
python3 finetune.py --model distilbert-base-multilingual-uncased --context long
python3 finetune.py --model distilbert-base-multilingual-uncased --context short
python3 finetune.py --model distilbert-base-multilingual-uncased --context graph

python3 finetune.py --model bert-base-multilingual-uncased --context None
python3 finetune.py --model bert-base-multilingual-uncased --context simple 
python3 finetune.py --model bert-base-multilingual-uncased --context long
python3 finetune.py --model bert-base-multilingual-uncased --context short
python3 finetune.py --model bert-base-multilingual-uncased --context graph

python3 finetune.py --model distilbert-base-multilingual-uncased --context None --validate
python3 finetune.py --model distilbert-base-multilingual-uncased --context simple --validate
python3 finetune.py --model distilbert-base-multilingual-uncased --context long --validate
python3 finetune.py --model distilbert-base-multilingual-uncased --context short --validate
python3 finetune.py --model distilbert-base-multilingual-uncased --context graph --validate

python3 finetune.py --model bert-base-multilingual-uncased --context None --validate
python3 finetune.py --model bert-base-multilingual-uncased --context simple --validate
python3 finetune.py --model bert-base-multilingual-uncased --context long --validate
python3 finetune.py --model bert-base-multilingual-uncased --context short --validate
python3 finetune.py --model bert-base-multilingual-uncased --context graph --validate

python3 finetune.py --model distilbert-base-multilingual-cased --context None --translate
python3 finetune.py --model distilbert-base-multilingual-cased --context simple --translate
python3 finetune.py --model distilbert-base-multilingual-cased --context long --translate
python3 finetune.py --model distilbert-base-multilingual-cased --context short --translate
python3 finetune.py --model distilbert-base-multilingual-cased --context graph --translate

python3 finetune.py --model bert-base-multilingual-cased --context None  --translate
python3 finetune.py --model bert-base-multilingual-cased --context simple --translate  
python3 finetune.py --model bert-base-multilingual-cased --context long --translate 
python3 finetune.py --model bert-base-multilingual-cased --context short --translate 
python3 finetune.py --model bert-base-multilingual-cased --context graph --translate 

python3 finetune.py --model distilbert-base-multilingual-cased --context None --validate --translate 
python3 finetune.py --model distilbert-base-multilingual-cased --context simple --validate --translate 
python3 finetune.py --model distilbert-base-multilingual-cased --context long --validate --translate 
python3 finetune.py --model distilbert-base-multilingual-cased --context short --validate --translate 
python3 finetune.py --model distilbert-base-multilingual-cased --context graph --validate --translate 

python3 finetune.py --model bert-base-multilingual-cased --context None --validate --translate 
python3 finetune.py --model bert-base-multilingual-cased --context simple --validate --translate 
python3 finetune.py --model bert-base-multilingual-cased --context long --validate --translate 
python3 finetune.py --model bert-base-multilingual-cased --context short --validate --translate 
python3 finetune.py --model bert-base-multilingual-cased --context graph --validate --translate 

python3 finetune.py --model distilgpt2 --translate --context None --zero_shot
python3 finetune.py --model distilgpt2 --translate --context simple --zero_shot
python3 finetune.py --model distilgpt2 --translate --context long --zero_shot
python3 finetune.py --model distilgpt2 --translate --context short --zero_shot
python3 finetune.py --model distilgpt2 --translate --context graph --zero_shot

python3 finetune.py --model gpt2 --translate --context None --zero_shot
python3 finetune.py --model gpt2 --translate --context simple --zero_shot
python3 finetune.py --model gpt2 --translate --context long --zero_shot
python3 finetune.py --model gpt2 --translate --context short --zero_shot
python3 finetune.py --model gpt2 --translate --context graph --zero_shot

