import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        # Defining a two layer MLP with tanh activation
        self.net = nn.Sequential(
            nn.Linear(21, 40),   # modify from 2 input/output to 3 input/output
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 21),
        )
        ## Providing a specific initialization of the weights and biases
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)