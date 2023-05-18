import sys
if __name__ == "__main__":
    sys.path.extend(['../'])
import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
import torch.nn as nn
from BNN.utils import *
import logging


class Dense(nn.Module):
    """
        Bayesian Fully connected layer
    """
    def __init__(self, in_size, out_size, device):
        """
            Parameters:
                in_size: size of input vector
                out_size: size of output vector
                deterministic: if set=True, the layer is initialized with N(0,I)
        """
        super(Dense, self).__init__()
        self.device = device
        if self.device == 'cuda':
            self.W_mean = nn.Parameter(torch.cuda.FloatTensor(size=(out_size, in_size)))
            self.W_std = nn.Parameter(torch.cuda.FloatTensor(size=(out_size, in_size)))
            self.b_mean = nn.Parameter(torch.cuda.FloatTensor(size=(1, out_size)))
            self.b_std = nn.Parameter(torch.cuda.FloatTensor(size=(1, out_size)))
        else:
            self.W_mean = nn.Parameter(torch.ones(size=(out_size, in_size)))
            self.W_std = nn.Parameter(torch.ones(size=(out_size, in_size)))
            self.b_mean = nn.Parameter(torch.ones(size=(1, out_size)))
            self.b_std = nn.Parameter(torch.ones(size=(1, out_size)))
        self.reset_parameters()

    def forward(self, X):
        # begin_time = time.time()
        if self.device == "cpu":
            W_dist = td.Normal(loc=self.W_mean, scale=F.softplus(self.W_std))
            W = W_dist.rsample()
            b_dist = td.Normal(loc=self.b_mean, scale=F.softplus(self.b_std))
            b = b_dist.rsample()
        else:
            W_eps = torch.cuda.FloatTensor(size=self.W_mean.shape).normal_()
            W = self.W_mean + W_eps*F.softplus(self.W_std)
            b_eps = torch.cuda.FloatTensor(size=self.b_mean.shape).normal_()
            b = self.b_mean + b_eps*F.softplus(self.b_std)
        output = F.linear(X, W, b)
        # print(f'a forward time: {time.time()-begin_time}')
        return output
    
    def broaden(self, diffusion):
        tmp_W = torch.log(
            torch.pow(1 + torch.exp(self.W_std.data), diffusion) - 1.)
        tmp_b = torch.log(
            torch.pow(1 + torch.exp(self.b_std.data), diffusion) - 1.)
        tmp_W[tmp_W == torch.inf] = self.W_std.data[tmp_W ==
                                                        torch.inf]*diffusion
        tmp_b[tmp_b == torch.inf] = self.b_std.data[tmp_b ==
                                                        torch.inf]*diffusion
        self.W_std.data = tmp_W
        self.b_std.data = tmp_b

    def log(self, logger):
        logger.info('\nDense***')
        logger.info(f'\tW_mean: {self.W_mean}')
        logger.info(f'\tW_std: {F.softplus(self.W_std)}')
        logger.info(f'\tb_mean: {self.b_mean}')
        logger.info(f'\tb_std: {F.softplus(self.b_std)}')

    def reset_parameters(self):
        def reset(mean, std):
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(mean)
            ratio = 0.5
            total_var = 2 / fan_in
            noise_var = total_var * ratio
            mu_var = total_var - noise_var
            noise_std, mu_std = np.sqrt(noise_var), np.sqrt(mu_var)
            bound = np.sqrt(3.0) * mu_std
            rho_init = np.log(np.exp(noise_std)-1)
            if self.device == "cuda":
                bound = torch.cuda.FloatTensor(np.asarray(bound))
                rho_init = torch.cuda.FloatTensor(np.asarray(rho_init))
                mean = (torch.cuda.FloatTensor(size=mean.size()).uniform_() - 0.5) * 2.0 * bound
                # std = torch.ones(size=std.size(), device="cuda")*rho_init
            else:
                nn.init.uniform_(mean, -bound, bound)
            nn.init.uniform_(std, rho_init, rho_init)
            return nn.Parameter(mean), nn.Parameter(std)

        self.W_mean, self.W_std = reset(self.W_mean, self.W_std)
        self.b_mean, self.b_std = reset(self.b_mean, self.b_std)

    @classmethod
    def kl_divergence(cls, layer_q, layer_p):
        assert isinstance(layer_p, Dense)
        assert isinstance(layer_q, Dense)

        q_W_std = F.softplus(layer_q.W_std)
        q_b_std = F.softplus(layer_q.b_std)
        p_W_std = F.softplus(layer_p.W_std)
        p_b_std = F.softplus(layer_p.b_std)

        kl_W = kl_Gauss_Gauss(layer_q.W_mean, q_W_std, layer_p.W_mean, p_W_std)
        kl_b = kl_Gauss_Gauss(layer_q.b_mean, q_b_std, layer_p.b_mean, p_b_std)
        return kl_W + kl_b


if __name__ == "__main__":
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler('log/Dense.log')
    file_handler.setFormatter((formatter))

    logger.addHandler(file_handler)
    layer = Dense(in_size=2, out_size=5, device='cpu')
    layer.log(logger)
    layer.broaden(2.)
    layer.log(logger)
    
