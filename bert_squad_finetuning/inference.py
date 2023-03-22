from transformers import BertForQuestionAnswering, BertTokenizerFast
from dataset import SquadDataset
import torch
from torch.utils.data import DataLoader

# load finetuned model

model_path = 'checkpoints'
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

device = 'cpu'

model = model.to(device)
model.eval()

context1 = """versions of the " doctor who theme " have also been released as pop music over the years. in the early 1970s, jon pertwee, who had played the 
third doctor, recorded a version of the doctor who theme with spoken lyrics, titled, " who is the doctor ". [ note 6 ] in 1978 a disco version 
of the theme was released in the uk, denmark and australia by the group mankind, which reached number 24 in the uk charts. in 1988 the band the 
justified ancients of mu mu ( later known as the klf ) released the single " doctorin ' the tardis " under the name the timelords, which reached 
no. 1 in the uk and no. 2 in australia ; this version incorporated several other songs, including " rock and roll part 2 " by gary glitter 
( who recorded vocals for some of the cd - single remix versions of " doctorin ' the tardis " ). others who have covered or reinterpreted the 
theme include orbital, pink floyd, the australian string ensemble fourplay, new zealand punk band blam blam blam, the pogues, thin lizzy, dub 
syndicate, and the comedians bill bailey and mitch benn. both the theme and obsessive fans were satirised on the chaser ' s war on everything. 
the theme tune has also appeared on many compilation cds, and has made its way into mobile - phone ringtones. fans have also produced and 
distributed their own remixes of the theme. in january 2011 the mankind version was released as a digital download on the album gallifrey and 
beyond."""# [SEP] which doctor who - related song reached number one in the uk?"""

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

def get_prediction(data):
    input_ids = data['input_ids']
    token_type_ids = data['token_type_ids']
    attention_mask = data['attention_mask']
    
    outputs = model(input_ids, token_type_ids, attention_mask)
    
    answer_start = torch.argmax(outputs[0])
    answer_end = torch.argmax(outputs[1])+1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(data['input_ids'][0][answer_start:answer_end]))
    
    return answer

def get_prediction_manual_question(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs[0])
    answer_end = torch.argmax(outputs[1])+1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    
    return answer

# load the test data dataset
testset = SquadDataset(train=False)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

for data in testloader:
    start_position = data['start_positions']
    end_postition = data['end_positions']
    true_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(data['input_ids'][0][start_position:end_postition]))
    pred_answer = get_prediction(data)
    info = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(data['input_ids'][0]))
    print(f'Question and context: {info}')
    print(f'True answer: {true_answer}')
    print(f'Predicted answer: {pred_answer}')

while 1:
    question = input('Please ask a question about the passage:')
    pred_answer = get_prediction_manual_question(context, question)
    print(f'Answer: {pred_answer}')