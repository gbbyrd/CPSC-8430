from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch
from torch.utils.data import DataLoader
from dataset import SquadDataset
from tqdm import tqdm

testset = SquadDataset(train=False)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

model=BertForQuestionAnswering.from_pretrained('bert-base-uncased')

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

model = model.to(device)

model.eval()

acc = []

for i in range(2):
  loop = tqdm(testloader, leave=True)
  for batch in loop:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answer_starts = batch['answer_starts'].to(device)
        answer_ends = batch['answer_ends'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        
        acc.append(((start_pred==answer_starts).sum()/len(start_pred)).item())
        acc.append(((end_pred==answer_ends).sum()/len(end_pred)).item())
  
  print("\n\nT/P\tanswer_start\tanswer_end\n")
  for i in range(len(answer_starts)):
    print(f"true\t{answer_starts[i]}\t{answer_ends[i]}\n"
          f"pred\t{start_pred[i]}\t{end_pred[i]}\n")
  print(f'Accuracy: {acc.sum()/len(acc)}')
  
  if i==0:
    print(f'\n\nInference: ---------------------------------------\n\n')
    model_path = 'checkpoints'
    model = BertForQuestionAnswering.from_pretrained(model_path)
