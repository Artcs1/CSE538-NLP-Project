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
