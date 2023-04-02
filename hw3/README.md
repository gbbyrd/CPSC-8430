# Homework 3 - Finetuning a Pre Trained BERT for Question Answering

## Finetuning
The huggingface accelerate library is used to speed up the finetuning and evalution
processes. Make sure you install all of the necessary dependencies before finetuning.

To finetune a model, make sure that you have the checkpoints/squad and checkpoints/spoken_squad
directories in your hw3 directory. This is where the models will be saved. You can finetune
any pretrained model using the following commands:

For training on SQuAD dataset
```accelerate launch squad_finetune.py --pretrained_model <model_name>```

For training on Spoken SQuAD datset
```accelerate launch spoken_finetune.py --pretrained_model <model_name>```

## Evaluation
To evaluate your models, you much specify the path locations for the models in the 
benchmark.py file. To run this script for my trained models, replace the entire checkpoints
folder found in this repository with the checkpoints folder that you download from
my google drive here:

https://drive.google.com/drive/folders/1UrnBzkBtn3nTI_iA_3ZDy-1netks0_-j?usp=share_link

Then run the following command to get the f1 and exact matches scores
```accelerate launch benchmark.py```