from dataset import SquadDataset

trainset = SquadDataset(train=True)

for i in range(10):
    print(trainset[i]['start_positions'])
    print(trainset[i]['end_positions'])