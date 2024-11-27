import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, depth, act):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = act 

        # input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x, final_act=False):
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        
        # last layer
        x = self.layers[-1](x) 
        if final_act == False:
            return x
        else:
            return torch.relu(x)
