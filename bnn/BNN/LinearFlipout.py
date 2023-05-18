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
# Linear Flipout Layers with flipout weight estimator to perform
# variational inference in Bayesian neural networks. Variational layers
# enables Monte Carlo approximation of the distribution over the weights
#
# @authors: Ranganath Krishnan, Piero Esposito
#
# ======================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from BNN.base_variational_layer import BaseVariationalLayer_
from BNN.utils import *

__all__ = ["LinearFlipout"]


class LinearFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.,
                 prior_rho=1.,
                 prior_mean_bias=0.,
                 prior_rho_bias=-3.0,
                 posterior_mu_init=0.,
                 posterior_rho_init=-3.0,
                 init_posterior_mean=None,
                 init_posterior_rho=None,
                 init_posterior_mean_bias=None,
                 init_posterior_rho_bias=None,
                 bias=True):
        """
        Implements Linear layer with Flipout reparameterization trick.
        Ref: https://arxiv.org/abs/1803.04386
        Inherits from bayesian_torch.layers.BaseVariationalLayer_
        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_rho: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias.   fault: True,
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.prior_mean = prior_mean
        self.prior_rho = prior_rho
        self.prior_mean_bias = prior_mean_bias
        self.prior_rho_bias = prior_rho_bias
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.init_posterior_mean = init_posterior_mean
        self.init_posterior_rho = init_posterior_rho
        self.init_posterior_mean_bias = init_posterior_mean_bias
        self.init_posterior_rho_bias = init_posterior_rho_bias

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu', torch.Tensor(
                out_features), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
            self.register_buffer('eps_bias', torch.Tensor(
                out_features), persistent=False)

        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            fan_in = self.prior_weight_mu.shape[1]
            fan_out = self.prior_weight_mu.shape[0]

            ratio = torch.tensor(0.5)
            total_var = torch.tensor(20 / fan_in)
            noise_var = total_var * ratio  # sigma_init^2
            mu_var = total_var - noise_var  # var[mu]

            noise_std = torch.sqrt(noise_var)
            mu_std = torch.sqrt(mu_var)
            bound = torch.sqrt(torch.tensor(3.)) * mu_std
            rho_init = torch.log(torch.expm1(noise_std))

            # prior weight mu
            if isinstance(self.prior_mean, float):
                self.prior_weight_mu.fill_(self.prior_mean)
            elif isinstance(self.prior_mean, torch.Tensor):
                self.prior_weight_mu.data = self.prior_mean
            else:
                self.prior_weight_mu.fill_(torch.tensor(0.))

            # prior weight sigma = softplus(prior_rho)
            if isinstance(self.prior_rho, float):
                self.prior_weight_sigma.fill_(
                    F.softplus(torch.tensor(self.prior_rho)))
            elif isinstance(self.prior_rho, torch.Tensor):
                self.prior_weight_sigma.data = F.softplus(self.prior_rho)
            else:
                self.prior_weight_sigma.fill_(torch.tensor(1./fan_in/0.334056))

            # init posterior weight mu
            if isinstance(self.init_posterior_mean, float):
                self.mu_weight.fill_(self.init_posterior_mean)
            elif isinstance(self.init_posterior_mean, torch.Tensor):
                self.mu_weight.data = self.init_posterior_mean
            else:
                self.mu_weight.data.uniform_(-bound, bound)

            # init posterior weight sigma
            if isinstance(self.init_posterior_rho, float):
                self.rho_weight.fill_(self.init_posterior_rho)
            elif isinstance(self.init_posterior_rho, torch.Tensor):
                self.rho_weight.data = self.init_posterior_rho
            else:
                self.rho_weight.fill_(rho_init)

            if self.mu_bias is not None:
                # prior bias mu
                if isinstance(self.prior_mean_bias, float):
                    self.prior_bias_mu.fill_(self.prior_mean_bias)
                elif isinstance(self.prior_mean_bias, torch.Tensor):
                    self.prior_bias_mu.data = self.prior_mean_bias
                else:
                    self.prior_bias_mu.data.fill_(torch.tensor(0.))

                # prior bias sigma = softplus(prior_rho_bias)
                if isinstance(self.prior_rho_bias, float):
                    self.prior_bias_sigma.fill_(F.softplus(
                        torch.tensor(self.prior_rho_bias)))
                elif isinstance(self.prior_rho_bias, torch.Tensor):
                    self.prior_bias_sigma.data = F.softplus(self.prior_rho_bias)
                else:
                    self.prior_bias_sigma.data.fill_(torch.tensor(1./0.334056/fan_in))

                # init posterior bias mu
                if isinstance(self.init_posterior_mean_bias, float):
                    self.mu_bias.fill_(self.init_posterior_mean_bias)
                elif isinstance(self.init_posterior_mean_bias, torch.Tensor):
                    self.mu_bias.data = self.init_posterior_mean_bias
                else:
                    self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)

                # init posterior bias sigma
                if isinstance(self.init_posterior_rho_bias, float):
                    self.rho_bias.fill_(self.init_posterior_rho_bias)
                elif isinstance(self.init_posterior_rho_bias, torch.Tensor):
                    self.rho_bias.data = self.init_posterior_rho_bias
                else:
                    self.rho_bias.data.normal_(
                        mean=self.posterior_rho_init, std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(self.mu_weight, sigma_weight,
                         self.prior_weight_mu, self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=False):
        if self.dnn_to_bnn_flag:
            return_kl = False
        # sampling delta_W
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        delta_weight = (sigma_weight * self.eps_weight.data.normal_())

        # get kl divergence
        if return_kl:
            kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # linear outputs
        outputs = F.linear(x, self.mu_weight, self.mu_bias)

        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        perturbed_outputs = F.linear(x * sign_input, delta_weight,
                                     bias) * sign_output

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
        assert (self.prior_weight_sigma > 0).all(
        ), "Prior sigma must be positive"
        self.rho_weight.data = inv_softplus(self.prior_weight_sigma)
        self.mu_weight.data = self.prior_weight_mu
        if self.bias:
            self.rho_bias.data = inv_softplus(self.prior_bias_sigma)
            self.mu_bias.data = self.prior_bias_mu
        assert (self.prior_weight_sigma > 0).all(
        ), "Prior sigma must be positive"

    @classmethod
    def kl_divergence(cls, layer_q, layer_p):
        q_sigma_weight = torch.log1p(torch.exp(layer_q.rho_weight))
        p_sigma_weight = torch.log1p(torch.exp(layer_p.rho_weight))
        kl = kl_Gauss_Gauss(layer_q.mu_weight, q_sigma_weight,
                            layer_p.mu_weight, p_sigma_weight)
        if layer_q.mu_bias is not None and layer_p.mu_bias is not None:
            q_sigma_bias = torch.log1p(torch.exp(layer_q.rho_bias))
            p_sigma_bias = torch.log1p(torch.exp(layer_p.rho_bias))
            kl += kl_Gauss_Gauss(layer_q.mu_bias, q_sigma_bias,
                                 layer_p.mu_bias, p_sigma_bias)
        elif layer_p.mu_bias is None and layer_q.mu_bias is None:
            pass
        else:
            raise ValueError('Both layers must have bias or not')
        return kl
