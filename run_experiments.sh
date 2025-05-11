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
