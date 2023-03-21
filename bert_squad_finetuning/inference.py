from dataset import SquadDataset

trainset = SquadDataset(train=True)

for i in range(10):
    data = trainset[i]
    print(data['input_ids'])
    print(data['token_type_ids'])
    print(data['attention_mask'])
    print(data['start_positions'])
    print(data['end_positions'])