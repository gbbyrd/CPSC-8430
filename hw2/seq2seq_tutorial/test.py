import torch
import random

arr = []
for  i in range(10):
    arr.append(random.random())
    
yes = torch.tensor(arr)

print(yes)

hm, hmm = yes.topk(1)

print(hm)
print(hmm)
hmm = hmm.squeeze()
print(hmm)
hmm = hmm.unsqueeze(0)
print(hmm)
hmm = hmm.unsqueeze(0)
print(hmm)