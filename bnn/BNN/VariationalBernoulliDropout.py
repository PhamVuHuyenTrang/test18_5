import sys
if __name__ == "__main__":
    sys.path.extend(['../'])

from torch import nn
import torch
import logging
from BNN.utils import *


class VariationalBernoulliDropout(nn.Module):
    def __init__(self,
                 in_features,
                 inv_sigmoid_prior_droprate=0.,
                 init_inv_sigmoid_posterior_droprate=None,
                 temperature=0.001):
        super(VariationalBernoulliDropout, self).__init__()
        self.in_features = in_features
        self.inv_sigmoid_droprate = nn.Parameter(torch.Tensor(in_features))
        self.inv_sigmoid_prior_droprate_0 = inv_sigmoid_prior_droprate
        self.init_inv_sigmoid_posterior_droprate = init_inv_sigmoid_posterior_droprate
        self.register_buffer('inv_sigmoid_prior_droprate',
                             torch.Tensor(in_features))
        self.temperature = temperature
        self.init_parameters()

    def init_parameters(self):
        # prior droprate
        if isinstance(self.inv_sigmoid_prior_droprate_0, float):
            self.inv_sigmoid_prior_droprate.data.fill_(
                self.inv_sigmoid_prior_droprate_0)
        elif isinstance(self.inv_sigmoid_prior_droprate_0, torch.Tensor):
            self.inv_sigmoid_prior_droprate.data = self.inv_sigmoid_prior_droprate_0
        else:
            self.inv_sigmoid_prior_droprate.data.normal_(0, 1.)

        # init posterior droprate
        if isinstance(self.init_inv_sigmoid_posterior_droprate, float):
            self.inv_sigmoid_droprate.data.fill_(
                self.init_inv_sigmoid_posterior_droprate)
        elif isinstance(self.init_inv_sigmoid_posterior_droprate, torch.Tensor):
            self.inv_sigmoid_droprate.data = self.init_inv_sigmoid_posterior_droprate
        else:
            self.inv_sigmoid_droprate.data = self.inv_sigmoid_prior_droprate

    def forward(self, X, st):
        droprates = torch.sigmoid(self.inv_sigmoid_droprate)
        # dropoutLogger = logging.getLogger('dropoutLogger')
        # dropoutLogger.info(f"droprates: {droprates}")
        if self.training and st == 1:
            # Gumbel softmax trick
            if len(X.shape) == 4:
                # image, X: (batch_size, channels, height, width)
                gumbel_mask = sample_gumbel_mask_for_image(X.shape, droprates,
                                                    temperature=self.temperature)
                mean_mask = droprate_to_mean_mask_image(X.shape, droprates).clamp(1e-5, 1.)
                mask = gumbel_mask / mean_mask
            else:
                gumbel_mask = sample_gumbel_mask((X.shape[0], X.shape[1]), droprates,
                                            temperature=self.temperature)
                mean_mask = droprate_to_mean_mask(X.shape, droprates).clamp(1e-5, 1.)
                mask = gumbel_mask / mean_mask
            return X * mask
        else:
            return X

    def kl_loss(self, st):
        if st:
            p = torch.sigmoid(self.inv_sigmoid_droprate)
            p = torch.clamp(p, 1e-5, 1. - 1e-5)
            prior_p = torch.sigmoid(self.inv_sigmoid_prior_droprate)
            prior_p = torch.clamp(prior_p, 1e-5, 1. - 1e-5)
            kl = p * torch.log(p/prior_p) \
                + (1. - p) * torch.log((1. - p)/(1. - prior_p))
            return kl.sum()
        else:
            return torch.tensor(0., device=self.inv_sigmoid_droprate.device)


if __name__ == "__main__":
    layer = VariationalBernoulliDropout(5, None, None)
    x = torch.randn(1, 5, 2, 2)
    print(x)
    x0 = layer(x, st=0)
    x1 = layer(x, st=1)
    print(torch.nn.functional.sigmoid(layer.inv_sigmoid_prior_droprate))
    layer.inv_sigmoid_droprate.data = layer.inv_sigmoid_prior_droprate*1.1
    print(torch.nn.functional.sigmoid(layer.inv_sigmoid_droprate))
    print(layer.kl_loss(st=1))
    # print((x0==x).all())
    # print(x1)
    # print(torch.nn.functional.sigmoid(layer.inv_sigmoid_droprate))