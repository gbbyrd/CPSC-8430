from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from transformers import AutoTokenizer, BertForQuestionAnswering
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--pretrained_model', action='store', type=str, required=True, help='pretrained model name from huggingface library')
arguments = parser.parse_args()

# load the squad dataset
raw_datasets = load_dataset("squad")

""" Format of hugging face dataset
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})
"""

model_checkpoint = arguments.pretrained_model

#############################################################################################
#                                   PREPROCESS THE DATA
#############################################################################################

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384
stride = 128

#################################### TRAINING ###############################################

def preprocess_training_examples(examples):
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

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

""" You can apply the above function to the entire training set using the Dataset.map() method
in the hugging face transformers library.
"""

train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

#############################################################################################
#                               custom pytorch training loop
#############################################################################################

from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW

train_dataset.set_format("torch")

trainloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
print(len(trainloader))

# Use hugging face accelerator to get all of the objects on the correct device? Not sure
# exactly how this works.

from accelerate import Accelerator
from tqdm.auto import tqdm
import torch

def main():
    train_dataset.set_format("torch")

    trainloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=8,
    )   

    model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    accelerator = Accelerator(mixed_precision='no')
    model, optimizer, trainloader = accelerator.prepare(
        model, optimizer, trainloader
    )

    from transformers import get_scheduler

    num_train_epochs = 3
    num_update_steps_per_epoch = len(trainloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # TRAINING LOOP

    progress_bar = tqdm(range(num_training_steps))

    output_dir = 'checkpoints/squad/' + arguments.pretrained_model
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for epoch in range(num_train_epochs):
        # TRAINING
        model.train()
        for step, batch in enumerate(trainloader):
            # model = model.to(device)
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(output_dir)
        
if __name__=='__main__':
    main()
