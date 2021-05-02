import torch
from torch import nn

class Prelim_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(Prelim_Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size).double(),
            nn.BatchNorm1d(hidden_size, affine=False).double(),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=.2),
            nn.Linear(hidden_size, hidden_size).double(),
            nn.BatchNorm1d(hidden_size, affine=False).double(),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=.2),
            nn.Linear(hidden_size, num_outputs).double(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

