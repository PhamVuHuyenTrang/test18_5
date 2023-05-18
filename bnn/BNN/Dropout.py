import sys
if __name__ == "__main__":
    sys.path.extend(['../'])
import torch
import torch.nn as nn
import sys
from BNN.utils import *
import time


class BernoulliDropout(nn.Module):
    def __init__(self, runlength, in_size, min_init_drop_rate_factors=0., max_init_drop_rate_factors=1.386, trainable=True, temperature=0.01, device='cpu'):
        super(BernoulliDropout, self).__init__()
        self.device = device
        if device == 'cuda':
            self.droprate_factors = nn.Parameter(torch.cuda.FloatTensor(size=(
                1, in_size)).uniform_(min_init_drop_rate_factors, max_init_drop_rate_factors))
        else:
            self.droprate_factors = nn.Parameter(torch.FloatTensor(size=(1, in_size)).uniform_(
                min_init_drop_rate_factors, max_init_drop_rate_factors))

        self.training = trainable
        self.temperature = temperature
        self.runlength = runlength
        self.in_size = in_size

    def forward(self, X):
        droprates = torch.sigmoid(self.droprate_factors)
        if self.training:
            mask = sample_mask(X.shape, droprates,
                               temperature=self.temperature)
            if self.runlength == 0:
                return X*mask
            else:
                return X
        else:
            if self.runlength == 0:
                return X * torch.subtract(torch.ones(self.in_size).to(self.device), droprates)
            else:
                return X

    def increase_runlength(self):
        self.runlength += 1

    @classmethod
    def kl_divergence(cls, layer_q, layer_p):
        assert isinstance(layer_q, BernoulliDropout)
        assert isinstance(layer_p, BernoulliDropout)

        if layer_q.runlength == 0:
            q = torch.sigmoid(layer_q.droprate_factors)
            p = torch.sigmoid(layer_p.droprate_factors).detach()
            return (q*torch.log(q/p) + (1.0-q)*torch.log((1.0-q)/(1.0-p))).sum()
        else:
            return 0.


if __name__ == "__main__":
    layer = BernoulliDropout(runlength=0)
    print(torch.sigmoid(layer.droprate_factors))
    x = torch.ones((2, 3))
    print(x)
    print(layer(x))
    # print(x)
    # print(layer(x))
