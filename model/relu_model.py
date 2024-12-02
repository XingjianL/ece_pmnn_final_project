import torch.nn as nn
import torch
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        # Defining a two layer MLP with tanh activation
        self.net = nn.Sequential(
            nn.Linear(21, 100),   # modify from 2 input/output to 3 input/output
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 13),
        )
        ## Providing a specific initialization of the weights and biases
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        #print(y.shape, t.shape,self.net(y).shape)
        return torch.cat([torch.zeros((y.shape[0],8)).cuda(), self.net(y)], 1)