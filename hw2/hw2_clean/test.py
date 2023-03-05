import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.lin = nn.Linear(10, 10)

    def forward(self,x):
        return self.lin(x)


if __name__=='__main__':
    mod = model()

    ModelSaveLoc = 'checkpoints'
    torch.save(mod, "{}/{}.h5".format(ModelSaveLoc, 'model0'))
