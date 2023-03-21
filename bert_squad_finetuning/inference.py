from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch

# load finetuned model

model_path = 'checkpoints'
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

device = 'cpu'

model = model.to(device)
model.eval()

def get_prediction(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs[0])
    answer_end = torch.argmax(outputs[1])+1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    
    return answer

# Read the 

while 1:
    question = input('Please ask a question about the passage:')
    pred_answer = get_prediction(context3, question)
    print(f'Answer: {pred_answer}')