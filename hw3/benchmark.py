import torch
from transformers import AutoTokenizer, BertForQuestionAnswering, default_data_collator
from datasets import load_dataset
import evaluate
import collections
import numpy as np
from dataset import SpokenSquadDataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

def preprocess_squad_validation_examples(examples):
    # preprocess the validation data
    model_checkpoint = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # data was trained using the below max length and stride
    max_length = 384
    stride = 128

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

# define a function to compute f1 score
def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    predicted_answers = []

    metric = evaluate.load("squad")
    
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def main():
    """Main evaluation loop
    """
    # load the squad dataset
    raw_datasets = load_dataset("squad")

    # creates the validation dataset using the above defined function
    squad_validation_dataset = raw_datasets["validation"].map(
        preprocess_squad_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    # create the spoken squad preprocessed validation dataset
    spoken_squad_validation_dataset = SpokenSquadDataset(train=False)
    
    # create the spoken squad unprocessed dataset. This will need to be used with
    # the compute metrics function later on
    spoken_squad_raw = SpokenSquadDataset(train=False, unprocessed=True)
    
    # define model checkpoint paths
    squad_model_checkpoint_path = 'checkpoints/squad'
    spoken_squad_model_checkpoint_path = 'checkpoints/spoken_squad'  
    
    # remove the example id and offset mapping columns as they are unnecessary
    # WARNING WARNING if you do not remove the offset mapping column you will get an
    # error when you try to convert to tensor related to some of the values in the list
    # being None type
    squad_validation_set = squad_validation_dataset.remove_columns(["example_id", "offset_mapping"])
    spoken_squad_validation_set = squad_validation_dataset.remove_columns(["example_id", "offset_mapping"])
    
    # set hugging face datasets to torch format
    squad_validation_set.set_format("torch")
    
    # set parameters
    squad_model = BertForQuestionAnswering.from_pretrained(squad_model_checkpoint_path)
    spoken_squad_model = BertForQuestionAnswering.from_pretrained(spoken_squad_model_checkpoint_path)
    
    # Create dataloaders
    squad_eval_loader = DataLoader(
        squad_validation_set,
        collate_fn=default_data_collator,
        batch_size=8,
    )

    spoken_squad_eval_loader = DataLoader(
        spoken_squad_validation_set,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=8
    )
    
    accelerator = Accelerator(mixed_precision='no')
    squad_model, spoken_squad_model, squad_eval_loader, spoken_squad_eval_loader = accelerator.prepare(
        squad_model, spoken_squad_model, squad_eval_loader, spoken_squad_eval_loader
    )
    
    """evaluate the performance of each model on each dataset"""
    
    # squad trained model on squad test dataset
    squad_model.eval()
    spoken_squad_model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(squad_eval_loader):
        with torch.no_grad():
            outputs = squad_model(**batch)
        
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(squad_validation_dataset)]
    end_logits = end_logits[: len(squad_validation_dataset)]
    
    metrics = compute_metrics(
        start_logits, end_logits, squad_validation_dataset, raw_datasets["validation"]
    )
    
    print(metrics)
    
    # spoken squad trained model on squad test dataset
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(squad_eval_loader):
        with torch.no_grad():
            outputs = spoken_squad_model(**batch)
        
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(squad_validation_dataset)]
    end_logits = end_logits[: len(squad_validation_dataset)]
    
    metrics = compute_metrics(
        start_logits, end_logits, squad_validation_dataset, raw_datasets["validation"]
    )
    
    print(metrics)
    
    # squad trained model on spoken squad test dataset
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(spoken_squad_eval_loader):
        with torch.no_grad():
            outputs = squad_model(**batch)
        
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(squad_validation_dataset)]
    end_logits = end_logits[: len(squad_validation_dataset)]
    
    metrics = compute_metrics(
        start_logits, end_logits, spoken_squad_validation_dataset, spoken_squad_raw
    )
    
    print(metrics)
    
    # spoken squad trained model on spoken squad test dataset
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(spoken_squad_eval_loader):
        with torch.no_grad():
            outputs = spoken_squad_model(**batch)
        
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(squad_validation_dataset)]
    end_logits = end_logits[: len(squad_validation_dataset)]
    
    metrics = compute_metrics(
        start_logits, end_logits, spoken_squad_validation_dataset, spoken_squad_raw
    )
    
    print(metrics)
    
if __name__=='__main__':
    main()
    
    
    


