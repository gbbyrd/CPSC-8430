from dataset import SpokenSquadDataset
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, BertForQuestionAnswering, get_scheduler
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm

# create spoken squad dataset and dataloader
trainset = SpokenSquadDataset()
trainloader = DataLoader(
    trainset, 
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8
)

# load model and choose optimizer
model_checkpoint = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
optimizer = AdamW(model.parameters(), lr=2e-5)

# utilize hugging face accelerator to ensure that all tensors are on the correct device
accelerator = Accelerator(mixed_precision='no')

model, optimizer, trainloader = accelerator.prepare(
    model, optimizer, trainloader
)

num_train_epochs = 3
num_update_steps_per_epoch = len(trainloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# train model
progress_bar = tqdm(range(num_training_steps))

output_dir = 'checkpoints/spoken_squad'

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(trainloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
# save the model to checkpoints directory
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
trainset.tokenizer.save_pretrained(output_dir)
