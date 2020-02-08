import torch
from torch import nn


class L2Normalization(nn.Module):
    """
    Author: Max deGroot : https://github.com/amdegroot/ssd.pytorch
    from https://github.com/amdegroot/ssd.pytorch ssd.pytorch/layers/modules/l2norm.py file
    """

    def __init__(self, input_channel, gamma=None, epsilon=1e-10):
        super(L2Normalization, self).__init__()
        self.input_channel = input_channel
        self.gamma = gamma
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.Tensor(self.input_channel))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.epsilon
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
