import requests
import json
import torch
import os
from tqdm import tqdm

# Load the training dataset and take a look at it
with open('data/squad1.1/train-v1.1.json', 'rb') as f:
  squad = json.load(f)
  
# Each 'data' dict has two keys (title and paragraphs)
squad['data'][0].keys()

# Find the group about Greece
gr = -1
for idx, group in enumerate(squad['data']):
  print(group['title'])
  if group['title'] == 'Greece':
    gr = idx
    print(gr)
    break

# let's check on Greece which is 186th (0-based indexing)
# we can see that we have a context and many questions and answers following
squad['data'][186]

# and this is the context given for NYC
squad['data'][186]['paragraphs'][0]['context']

def read_data(path):  
  # load the json file
  with open(path, 'rb') as f:
    squad = json.load(f)

  contexts = []
  questions = []
  answers = []

  for group in squad['data']:
    for passage in group['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
        question = qa['question']
        for answer in qa['answers']:
          contexts.append(context)
          questions.append(question)
          answers.append(answer)

  return contexts, questions, answers

train_contexts, train_questions, train_answers = read_data('data/squad1.1/train-v1.1.json')
valid_contexts, valid_questions, valid_answers = read_data('data/squad1.1/dev-v1.1.json')

# print a random question and answer
print(f'There are {len(train_questions)} questions')
print(train_questions[-10000])
print(train_answers[-10000])

def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two so we fix this
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)

# You can see that now we get the answer_end also
print(train_questions[-10000])
print(train_answers[-10000])

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)

train_encodings.keys()

no_of_encodings = len(train_encodings['input_ids'])
print(f'We have {no_of_encodings} context-question pairs')

train_encodings['input_ids'][0]

tokenizer.decode(train_encodings['input_ids'][0])

def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)

train_encodings['start_positions'][:10]

class SQuAD_Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)

train_dataset = SQuAD_Dataset(train_encodings)
valid_dataset = SQuAD_Dataset(valid_encodings)

from torch.utils.data import DataLoader

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Check on the available device - use GPU
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

print(f'Working on {device}')

from transformers import AdamW

N_EPOCHS = 3
optim = AdamW(model.parameters(), lr=5e-5)

# model.to(device)
# model.train()

# for epoch in range(N_EPOCHS):
#   loop = tqdm(train_loader, leave=True)
#   for batch in loop:
#     optim.zero_grad()
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     start_positions = batch['start_positions'].to(device)
#     end_positions = batch['end_positions'].to(device)
#     outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
#     loss = outputs[0]
#     loss.backward()
#     optim.step()

#     loop.set_description(f'Epoch {epoch+1}')
#     loop.set_postfix(loss=loss.item())
    
model_path = 'checkpoints'
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)

# Load the model for testing

model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

model = model.to(device)

model.eval()

acc = []

for batch in tqdm(valid_loader):
  with torch.no_grad():
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_true = batch['start_positions'].to(device)
    end_true = batch['end_positions'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)

    start_pred = torch.argmax(outputs['start_logits'], dim=1)
    end_pred = torch.argmax(outputs['end_logits'], dim=1)

    acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
    acc.append(((end_pred == end_true).sum()/len(end_pred)).item())

acc = sum(acc)/len(acc)

print("\n\nT/P\tanswer_start\tanswer_end\n")
for i in range(len(start_true)):
  print(f"true\t{start_true[i]}\t{end_true[i]}\n"
        f"pred\t{start_pred[i]}\t{end_pred[i]}\n")
  
def get_prediction(context, question):
  inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
  outputs = model(**inputs)
  
  answer_start = torch.argmax(outputs[0])  
  answer_end = torch.argmax(outputs[1]) + 1 
  
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
  
  return answer

def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re
  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)
  def white_space_fix(text):
    return " ".join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()
  
  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)
  
  common_tokens = set(pred_tokens) & set(truth_tokens)
  
  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0
  
  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)
  
  return round(2 * (prec * rec) / (prec + rec), 2)
  
def question_answer(context, question,answer):
  prediction = get_prediction(context,question)
  em_score = exact_match(prediction, answer)
  f1_score = compute_f1(prediction, answer)

  print(f'Question: {question}')
  print(f'Prediction: {prediction}')
  print(f'True Answer: {answer}')
  print(f'Exact match: {em_score}')
  print(f'F1 score: {f1_score}\n')
  
context = """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, 
        songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing 
        and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. 
        Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. 
        Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, 
        earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""


questions = ["For whom the passage is talking about?",
             "When did Beyonce born?",
             "Where did Beyonce born?",
             "What is Beyonce's nationality?",
             "Who was the Destiny's group manager?",
             "What name has the Beyoncé's debut album?",
             "How many Grammy Awards did Beyonce earn?",
             "When did the Beyoncé's debut album release?",
             "Who was the lead singer of R&B girl-group Destiny's Child?"]

answers = ["Beyonce Giselle Knowles - Carter", "September 4, 1981", "Houston, Texas", 
           "American", "Mathew Knowles", "Dangerously in Love", "five", "2003", 
           "Beyonce Giselle Knowles - Carter"]

for question, answer in zip(questions, answers):
  question_answer(context, question, answer)
  
context = """Athens is the capital and largest city of Greece. Athens dominates the Attica region and is one of the world's oldest cities, 
             with its recorded history spanning over 3,400 years and its earliest human presence starting somewhere between the 11th and 7th millennium BC.
             Classical Athens was a powerful city-state. It was a center for the arts, learning and philosophy, and the home of Plato's Academy and Aristotle's Lyceum.
             It is widely referred to as the cradle of Western civilization and the birthplace of democracy, largely because of its cultural and political impact on the European continent—particularly Ancient Rome.
             In modern times, Athens is a large cosmopolitan metropolis and central to economic, financial, industrial, maritime, political and cultural life in Greece. 
             In 2021, Athens' urban area hosted more than three and a half million people, which is around 35% of the entire population of Greece.
             Athens is a Beta global city according to the Globalization and World Cities Research Network, and is one of the biggest economic centers in Southeastern Europe. 
             It also has a large financial sector, and its port Piraeus is both the largest passenger port in Europe, and the second largest in the world."""

questions = ["Which is the largest city in Greece?",
             "For what was the Athens center?",
             "Which city was the home of Plato's Academy?"]

answers = ["Athens", "center for the arts, learning and philosophy", "Athens"]

for question, answer in zip(questions, answers):
  question_answer(context, question, answer)
  
context = """Angelos Poulis was born on 8 April 2001 in Nicosia, Cyprus. He is half Cypriot and half Greek. 
            He is currently studying at the Department of Informatics and Telecommunications of the University of Athens in Greece. 
            His scientific interests are in the broad field of Artificial Intelligence and he loves to train neural networks! 
            Okay, I'm Angelos and I'll stop talking about me right now."""

questions = ["When did Angelos born?",
             "In what university is Angelos studying now?",
             "What is Angelos' nationality?",
             "What are his scientific interests?",
             "What I will do right now?"]

answers = ["8 April 2001", "University of Athens", 
           "half Cypriot and half Greek", "Artificial Intelligence", 
           "stop talking about me"]

for question, answer in zip(questions, answers):
  question_answer(context, question, answer)
