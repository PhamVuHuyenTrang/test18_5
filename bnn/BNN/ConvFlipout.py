# Copyright (C) 2021 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Convolutional layers with flipout Monte Carlo weight estimator to perform
# variational inference in Bayesian neural networks. Variational layers
# enables Monte Carlo approximation of the distribution over the kernel
#
# @authors: Ranganath Krishnan, Piero Esposito
#
# https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/layers/base_variational_layer.py
# added broaden for conv2dflipout
#
# ======================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from BNN.base_variational_layer import BaseVariationalLayer_, get_kernel_size
from BNN.utils import kl_Gauss_Gauss, inv_softplus

from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

__all__ = [
    'Conv2dFlipout',
]


class Conv2dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0.,
                 prior_rho=-3.,
                 prior_mean_bias=0.,
                 prior_rho_bias=-3.,
                 posterior_mu_init=0.,
                 posterior_rho_init=-3.0,
                 init_posterior_mean=None,
                 init_posterior_rho=None,
                 init_posterior_mean_bias=None,
                 init_posterior_rho_bias=None,
                 bias=True):
        """
        Implements Conv2d layer with Flipout reparameterization trick.
        Inherits from bayesian_torch.layers.BaseVariationalLayer_
        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
                        or torch.tensor -> mean of the prior distribution, set a mean for each weight,
            prior_rho: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
                        or torch.tensor -> variance of the prior distribution, set a variance for each weight,
            prior_mean_bias: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
                             or torch.tensor -> mean of the prior distribution, set a mean for each bias,
            prior_rho_bias: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
                                 or torch.tensor -> variance of the prior distribution, set a variance for each bias,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.prior_mean = prior_mean
        self.prior_rho = prior_rho
        self.prior_mean_bias = prior_mean_bias
        self.prior_rho_bias = prior_rho_bias
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias
        self.init_posterior_mean = init_posterior_mean
        self.init_posterior_rho = init_posterior_rho
        self.init_posterior_mean_bias = init_posterior_mean_bias
        self.init_posterior_rho_bias = init_posterior_rho_bias

        self.kl = 0
        kernel_size = get_kernel_size(kernel_size, 2)
        self.mu_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(
                out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(
                out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            receptive_field_size = self.prior_weight_mu[0][0].numel()

            fan_in = self.in_channels * receptive_field_size
            fan_out = self.out_channels * receptive_field_size
            
            ratio = torch.tensor(0.5)
            total_var = torch.tensor(2. / fan_out)
            noise_var = total_var * ratio
            mu_var = total_var - noise_var

            noise_std = torch.sqrt(noise_var)
            mu_std = torch.sqrt(mu_var)
            bound = torch.sqrt(torch.tensor(3.)) * mu_std
            rho_init = torch.log(torch.exp(noise_std) - 1)  # sigma_init rho
            
            # prior weight mu
            if isinstance(self.prior_mean, float):
                self.prior_weight_mu.data.fill_(self.prior_mean)
            elif isinstance(self.prior_mean, torch.Tensor):
                self.prior_weight_mu.data = self.prior_mean
            else:
                self.prior_weight_mu.fill_(torch.tensor(0.))

            # prior weight sigma = softplus(prior_rho)
            if isinstance(self.prior_rho, float):
                self.prior_weight_sigma.data.fill_(
                    F.softplus(torch.tensor(self.prior_rho)))
            elif isinstance(self.prior_rho, torch.Tensor):
                self.prior_weight_sigma.data = F.softplus(self.prior_rho)
            else:
                self.prior_weight_sigma.data.fill_(torch.tensor(1./0.334056/fan_in))

            # init our weights for the deterministic and perturbated weights
            # init posterior kernel mu
            if self.init_posterior_mean is not None:
                self.mu_kernel.data = self.init_posterior_mean
            else:
                self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)

            # init posterior kernel rho
            if self.init_posterior_rho is not None:
                self.rho_kernel.data = self.init_posterior_rho
            else:
                self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

            if self.bias:
                # init posterior bias mu
                if self.init_posterior_mean_bias is not None:
                    self.mu_bias.data = self.init_posterior_mean_bias
                else:
                    self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)

                # init posterior bias rho
                if self.init_posterior_rho_bias is not None:
                    self.rho_bias.data = self.init_posterior_rho_bias
                else:
                    self.rho_bias.data.normal_(
                        mean=self.posterior_rho_init, std=0.1)

                # prior bias mu
                if isinstance(self.prior_mean_bias, float):
                    self.prior_bias_mu.data.fill_(self.prior_mean_bias)
                elif isinstance(self.prior_mean_bias, torch.Tensor):
                    self.prior_bias_mu.data = self.prior_mean_bias
                else:
                    self.prior_bias_mu.data.fill_(torch.tensor(0.))

                # prior bias sigma = softplus(prior_rho_bias)
                if isinstance(self.prior_rho_bias, float):
                    self.prior_bias_sigma.data.fill_(
                        F.softplus(torch.tensor(self.prior_rho_bias)))
                elif isinstance(self.prior_rho_bias, torch.Tensor):
                    self.prior_bias_sigma.data = F.softplus(self.prior_rho_bias)
                else:
                    self.prior_bias_sigma.fill_(torch.tensor(1./0.334056/fan_in))

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight,
                         self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=False):
        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv2d(x,
                           weight=self.mu_kernel,
                           bias=self.mu_bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        perturbed_outputs = F.conv2d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups) * sign_output

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs

    def broaden(self, diffusion):
        self.prior_weight_sigma.data = self.prior_weight_sigma.data * diffusion
        if self.bias:
            self.prior_bias_sigma.data = self.prior_bias_sigma.data * diffusion
        self.make_init_posterior_same_as_prior()

    def make_init_posterior_same_as_prior(self):
        self.rho_kernel.data = inv_softplus(self.prior_weight_sigma)
        self.mu_kernel.data = self.prior_weight_mu
        self.rho_bias.data = inv_softplus(self.prior_bias_sigma).view(-1)
        self.mu_bias.data = self.prior_bias_mu

    @classmethod
    def kl_divergence(cls, layer_q, layer_p):
        q_sigma_kernel = torch.log1p(torch.exp(layer_q.rho_kernel))
        p_sigma_kernel = torch.log1p(torch.exp(layer_p.rho_kernel))
        kl = kl_Gauss_Gauss(layer_q.mu_kernel, q_sigma_kernel,
                            layer_p.mu_kernel, p_sigma_kernel)
        if layer_q.bias and layer_p.bias:
            q_sigma_bias = torch.log1p(torch.exp(layer_q.rho_bias))
            p_sigma_bias = torch.log1p(torch.exp(layer_p.rho_bias))
            kl += kl_Gauss_Gauss(layer_q.mu_bias, q_sigma_bias,
                                 layer_p.mu_bias, p_sigma_bias)
        elif not layer_p.bias and not layer_q.bias:
            pass
        else:
            raise ValueError('Both layers must have bias or not')
        return kl
