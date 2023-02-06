import torch
import torch.nn as nn
from sklearn.decomposition import PCA

class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 500)
        self.fc3 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    
if __name__ == '__main__':
    network = Network()
    
    first_layer_params = ...
    for count, (name, param) in enumerate(network.named_parameters()):
        if count == 1:
            break
        first_layer_params = param.view(-1)
    
    new_tensor = torch.tensor([[1, 2, 3, 4],
                              [5, 6, 7, 8]])
    print(new_tensor.size())
    pca = PCA(n_components=2)
    pca_new_tensor = pca.fit_transform(new_tensor)
    print(pca_new_tensor.shape())
    yes = 0