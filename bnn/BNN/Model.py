import sys
if __name__ == "__main__":
    sys.path.extend(['../'])
import torch
import torch.nn as nn
from copy import deepcopy
from BNN.LinearFlipout import LinearFlipout
from BNN.Dense import Dense
from BNN.Dropout import BernoulliDropout
from BNN.VariationalBernoulliDropout import VariationalBernoulliDropout
from BNN.VariationalGaussianDropout import GaussDropout, GaussDropoutConv2d
from BNN.LinearFlipout import LinearFlipout
from BNN.ConvFlipout import Conv2dFlipout
import copy
import logging


class Model(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(Model, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 20
        else:
            prior = list(prior)

        if init_posterior is None:
            init_posterior = [None] * 20

        self.conv1 = Conv2dFlipout(
            in_channels=3, out_channels=32, kernel_size=3,
            prior_mean=prior[0], prior_rho=prior[1],
            prior_mean_bias=prior[2], prior_rho_bias=prior[3],
            init_posterior_mean=init_posterior[0], init_posterior_rho=init_posterior[1],
            init_posterior_mean_bias=init_posterior[2], init_posterior_rho_bias=init_posterior[3]).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.conv2 = Conv2dFlipout(
            in_channels=32, out_channels=32, kernel_size=3,
            prior_mean=prior[4], prior_rho=prior[5],
            prior_mean_bias=prior[6], prior_rho_bias=prior[7],
            init_posterior_mean=init_posterior[4], init_posterior_rho=init_posterior[5],
            init_posterior_mean_bias=init_posterior[6], init_posterior_rho_bias=init_posterior[7]).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.dropout1 = nn.Dropout(p=0.2).to(device)
        self.conv3 = Conv2dFlipout(
            in_channels=32, out_channels=64, kernel_size=3,
            prior_mean=prior[8], prior_rho=prior[9],
            prior_mean_bias=prior[10], prior_rho_bias=prior[11],
            init_posterior_mean=init_posterior[8], init_posterior_rho=init_posterior[9],
            init_posterior_mean_bias=init_posterior[10], init_posterior_rho_bias=init_posterior[11]).to(device)
        self.relu3 = nn.ReLU().to(device)
        self.conv4 = Conv2dFlipout(
            in_channels=64, out_channels=64, kernel_size=3,
            prior_mean=prior[12], prior_rho=prior[13],
            prior_mean_bias=prior[14], prior_rho_bias=prior[15],
            init_posterior_mean=init_posterior[12], init_posterior_rho=init_posterior[13],
            init_posterior_mean_bias=init_posterior[14], init_posterior_rho_bias=init_posterior[15]).to(device)
        self.relu4 = nn.ReLU().to(device)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.dropout2 = nn.Dropout(p=0.2).to(device)
        self.flatten = nn.Flatten().to(device)
        self.dense = LinearFlipout(
            in_features=1600, out_features=10,
            prior_mean=prior[16], prior_rho=prior[17],
            prior_mean_bias=prior[18], prior_rho_bias=prior[19],
            init_posterior_mean=init_posterior[16], init_posterior_rho=init_posterior[17],
            init_posterior_mean_bias=init_posterior[18], init_posterior_rho_bias=init_posterior[19]).to(device)

    def forward(self, X):
        output = X
        for layer in self.layers():
            output = layer(output)
        return output

    def layers(self):
        return [self.conv1,
                self.relu1,
                self.conv2,
                self.relu2,
                self.maxpool1,
                self.dropout1,
                self.conv3,
                self.relu3,
                self.conv4,
                self.relu4,
                self.maxpool2,
                self.dropout2,
                self.flatten,
                self.dense]

    def broaden(self, diffusion):
        self.set_trainable(False)
        diffusion = torch.Tensor([[diffusion]]).to(self.device)
        for layer in self.layers():
            if isinstance(layer, LinearFlipout):
                layer.broaden(diffusion)
            if isinstance(layer, Conv2dFlipout):
                layer.broaden(diffusion)
        self.set_trainable(True)

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, Model)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = [statedict['conv1.mu_kernel'], statedict['conv1.rho_kernel'],
                 statedict['conv1.mu_bias'], statedict['conv1.rho_bias'],
                 statedict['conv2.mu_kernel'], statedict['conv2.rho_kernel'],
                 statedict['conv2.mu_bias'], statedict['conv2.rho_bias'],
                 statedict['conv3.mu_kernel'], statedict['conv3.rho_kernel'],
                 statedict['conv3.mu_bias'], statedict['conv3.rho_bias'],
                 statedict['conv4.mu_kernel'], statedict['conv4.rho_kernel'],
                 statedict['conv4.mu_bias'], statedict['conv4.rho_bias'],
                 statedict['dense.mu_weight'], statedict['dense.rho_weight'],
                 statedict['dense.mu_bias'], statedict['dense.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    @classmethod
    def from_dnn_prior(cls, path, model):
        state_dict = torch.load(path)
        prior = [state_dict['conv1.weight'], None,
                 state_dict['conv1.bias'], None,
                 state_dict['conv2.weight'], None,
                 state_dict['conv2.bias'],None,
                 state_dict['conv3.weight'], None,
                 state_dict['conv3.bias'], None,
                 state_dict['conv4.weight'], None,
                 state_dict['conv4.bias'],None,
                 state_dict['dense.weight'], None,
                 state_dict['dense.bias'],None]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    @classmethod
    def kl_divergence(cls, Model_q, Model_p):
        assert 'not to use!'
        assert 'not to use!'
        assert isinstance(Model_q, Model)
        assert isinstance(Model_p, Model)

        res = 0.0
        for (layer_q, layer_p) in zip(Model_q.layers(), Model_p.layers()):
            if isinstance(layer_q, BernoulliDropout):
                kl_dropout = BernoulliDropout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dropout * kl_dropout
            elif isinstance(layer_q, Dense):
                kl_dense = Dense.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, LinearFlipout):
                kl_dense = LinearFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, Conv2dFlipout):
                kl_dense = Conv2dFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
        return res

    def kl(self, st):
        kl = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
            elif isinstance(layer, LinearFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
        return kl, kl / self.weight_kl_dense, torch.tensor(0.).to(self.device)

    @classmethod
    def get_surrogate_prior(cls, path, model):
        state_dict = torch.load(path)
        prior = [state_dict['conv1.mu_kernel'], state_dict['conv1.rho_kernel'],
                 state_dict['conv1.mu_bias'], state_dict['conv1.rho_bias'],
                 state_dict['conv2.mu_kernel'], state_dict['conv2.rho_kernel'],
                 state_dict['conv2.mu_bias'], state_dict['conv2.rho_bias'],
                 state_dict['conv3.mu_kernel'], state_dict['conv3.rho_kernel'],
                 state_dict['conv3.mu_bias'], state_dict['conv3.rho_bias'],
                 state_dict['conv4.mu_kernel'], state_dict['conv4.rho_kernel'],
                 state_dict['conv4.mu_bias'], state_dict['conv4.rho_bias'],
                 state_dict['dense.mu_weight'], state_dict['dense.rho_weight'],
                 state_dict['dense.mu_bias'], state_dict['dense.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class DropoutModel(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st / runlength
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(DropoutModel, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 25

        if init_posterior is None:
            init_posterior = [None] * 25

        # self.dropout1 = VariationalBernoulliDropout(
        #     in_features=3, inv_sigmoid_prior_droprate=prior[0],
        #     init_inv_sigmoid_posterior_droprate=init_posterior[0]).to(device)
        self.conv1 = Conv2dFlipout(
            in_channels=3, out_channels=32, kernel_size=3,
            prior_mean=prior[1], prior_rho=prior[2],
            prior_mean_bias=prior[3], prior_rho_bias=prior[4],
            init_posterior_mean=init_posterior[1], init_posterior_rho=init_posterior[2],
            init_posterior_mean_bias=init_posterior[3], init_posterior_rho_bias=init_posterior[4]).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.dropout2 = VariationalBernoulliDropout(
            in_features=32, inv_sigmoid_prior_droprate=prior[5],
            init_inv_sigmoid_posterior_droprate=init_posterior[5]).to(device)
        self.conv2 = Conv2dFlipout(
            in_channels=32, out_channels=32, kernel_size=3,
            prior_mean=prior[6], prior_rho=prior[7],
            prior_mean_bias=prior[8], prior_rho_bias=prior[9],
            init_posterior_mean=init_posterior[6], init_posterior_rho=init_posterior[7],
            init_posterior_mean_bias=init_posterior[8], init_posterior_rho_bias=init_posterior[9]).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.dropout3 = VariationalBernoulliDropout(
            in_features=32, inv_sigmoid_prior_droprate=prior[10],
            init_inv_sigmoid_posterior_droprate=init_posterior[10]).to(device)
        self.conv3 = Conv2dFlipout(
            in_channels=32, out_channels=64, kernel_size=3,
            prior_mean=prior[11], prior_rho=prior[12],
            prior_mean_bias=prior[13], prior_rho_bias=prior[14],
            init_posterior_mean=init_posterior[11], init_posterior_rho=init_posterior[12],
            init_posterior_mean_bias=init_posterior[13], init_posterior_rho_bias=init_posterior[14]).to(device)
        self.relu3 = nn.ReLU().to(device)
        self.dropout4 = VariationalBernoulliDropout(
            in_features=64, inv_sigmoid_prior_droprate=prior[15],
            init_inv_sigmoid_posterior_droprate=init_posterior[15]).to(device)
        self.conv4 = Conv2dFlipout(
            in_channels=64, out_channels=64, kernel_size=3,
            prior_mean=prior[16], prior_rho=prior[17],
            prior_mean_bias=prior[18], prior_rho_bias=prior[19],
            init_posterior_mean=init_posterior[16], init_posterior_rho=init_posterior[17],
            init_posterior_mean_bias=init_posterior[18], init_posterior_rho_bias=init_posterior[19]).to(device)
        self.relu4 = nn.ReLU().to(device)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.dropout5 = VariationalBernoulliDropout(
            in_features=64, inv_sigmoid_prior_droprate=prior[20],
            init_inv_sigmoid_posterior_droprate=init_posterior[20]).to(device)
        self.flatten = nn.Flatten().to(device)
        self.dense = LinearFlipout(
            in_features=1600, out_features=10,
            prior_mean=prior[21], prior_rho=prior[22],
            prior_mean_bias=prior[23], prior_rho_bias=prior[24],
            init_posterior_mean=init_posterior[21], init_posterior_rho=init_posterior[22],
            init_posterior_mean_bias=init_posterior[23], init_posterior_rho_bias=init_posterior[24]).to(device)

    def forward(self, X, st=0):
        output = X
        for layer in self.layers():
            if isinstance(layer, VariationalBernoulliDropout):
                output = layer(output, st)
            else:
                output = layer(output)
        return output

    def layers(self):
        return [
                self.conv1,
                self.relu1,
                self.dropout2,
                self.conv2,
                self.relu2,
                self.maxpool1,
                self.dropout3,
                self.conv3,
                self.relu3,
                self.dropout4,
                self.conv4,
                self.relu4,
                self.maxpool2,
                self.dropout5,
                self.flatten,
                self.dense]

    def broaden(self, diffusion):
        pass

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def pertube(self):
        assert 'not to use!'
        if self.config.droprate_init_strategy == 0:
            pass
        elif self.config.droprate_init_strategy == 1:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.rand_like(layer.droprate_factor)
                    layer.droprate_factor.data = layer.droprate_factor.data * eps
        elif self.config.droprate_init_strategy == 2:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.normal(mean=0.0, std=torch.Tensor(
                        [1.0]), device=layer.device)
                    layer.droprate_factor.data = layer.droprate_factor.data + eps

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, DropoutModel)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = [None,
                 statedict['conv1.mu_kernel'], statedict['conv1.rho_kernel'],
                 statedict['conv1.mu_bias'], statedict['conv1.rho_bias'],
                 statedict['dropout2.inv_sigmoid_droprate'],
                 statedict['conv2.mu_kernel'], statedict['conv2.rho_kernel'],
                 statedict['conv2.mu_bias'], statedict['conv2.rho_bias'],
                 statedict['dropout3.inv_sigmoid_droprate'],
                 statedict['conv3.mu_kernel'], statedict['conv3.rho_kernel'],
                 statedict['conv3.mu_bias'], statedict['conv3.rho_bias'],
                 statedict['dropout4.inv_sigmoid_droprate'],
                 statedict['conv4.mu_kernel'], statedict['conv4.rho_kernel'],
                 statedict['conv4.mu_bias'], statedict['conv4.rho_bias'],
                 statedict['dropout5.inv_sigmoid_droprate'],
                 statedict['dense.mu_weight'], statedict['dense.rho_weight'],
                 statedict['dense.mu_bias'], statedict['dense.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model
    
    @classmethod
    def from_dnn_prior(cls, path, model):
        statedict = torch.load(path)
        prior = [None,
                 statedict['conv1.weight'], None,
                 statedict['conv1.bias'], None,
                 None,
                 statedict['conv2.weight'], None,
                 statedict['conv2.bias'], None,
                 None,
                 statedict['conv3.weight'], None,
                 statedict['conv3.bias'], None,
                 None,
                 statedict['conv4.weight'], None,
                 statedict['conv4.bias'], None,
                 None,
                 statedict['dense.weight'], None,
                 statedict['dense.bias'], None]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    @classmethod
    def kl_divergence(cls, Model_q, Model_p):
        assert 'not to use!'
        assert isinstance(Model_q, Model)
        assert isinstance(Model_p, Model)

        res = 0.0
        for (layer_q, layer_p) in zip(Model_q.layers(), Model_p.layers()):
            if isinstance(layer_q, BernoulliDropout):
                kl_dropout = BernoulliDropout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dropout * kl_dropout
            elif isinstance(layer_q, Dense):
                kl_dense = Dense.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, LinearFlipout):
                kl_dense = LinearFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, Conv2dFlipout):
                kl_dense = Conv2dFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
        return res

    def kl(self, st):
        kl = torch.tensor(0., device=self.device)
        kl_gauss = torch.tensor(0., device=self.device)
        kl_dropout = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, LinearFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, VariationalBernoulliDropout):
                kl_tmp = layer.kl_loss(st)
                kl += kl_tmp * self.weight_kl_dropout
                kl_dropout += kl_tmp
        return kl, kl_gauss, kl_dropout

    @classmethod
    def get_surrogate_prior(cls, path, model):
        state_dict = torch.load(path)
        prior = [None,
                 state_dict['conv1.mu_kernel'], state_dict['conv1.rho_kernel'],
                 state_dict['conv1.mu_bias'], state_dict['conv1.rho_bias'],
                 state_dict['dropout2.inv_sigmoid_droprate'],
                 state_dict['conv2.mu_kernel'], state_dict['conv2.rho_kernel'],
                 state_dict['conv2.mu_bias'], state_dict['conv2.rho_bias'],
                 state_dict['dropout3.inv_sigmoid_droprate'],
                 state_dict['conv3.mu_kernel'], state_dict['conv3.rho_kernel'],
                 state_dict['conv3.mu_bias'], state_dict['conv3.rho_bias'],
                 state_dict['dropout4.inv_sigmoid_droprate'],
                 state_dict['conv4.mu_kernel'], state_dict['conv4.rho_kernel'],
                 state_dict['conv4.mu_bias'], state_dict['conv4.rho_bias'],
                 state_dict['dropout5.inv_sigmoid_droprate'],
                 state_dict['dense.mu_weight'], state_dict['dense.rho_weight'],
                 state_dict['dense.mu_bias'], state_dict['dense.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class ModelDNN(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(ModelDNN, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [0., 1.] * 10
        else:
            prior = list(prior)

        if init_posterior is None:
            init_posterior = [None] * 20

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.dropout1 = nn.Dropout(p=0.2).to(device)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3).to(device)
        self.relu3 = nn.ReLU().to(device)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3).to(device)
        self.relu4 = nn.ReLU().to(device)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.dropout2 = nn.Dropout(p=0.2).to(device)
        self.flatten = nn.Flatten().to(device)
        self.dense = nn.Linear(in_features=1600, out_features=10).to(device)

    def forward(self, X):
        output = X
        for layer in self.layers():
            output = layer(output)
        return output

    def layers(self):
        return [self.conv1,
                self.relu1,
                self.conv2,
                self.relu2,
                self.maxpool1,
                self.dropout1,
                self.conv3,
                self.relu3,
                self.conv4,
                self.relu4,
                self.maxpool2,
                self.dropout2,
                self.flatten,
                self.dense]

    def broaden(self, diffusion):
        self.set_trainable(False)
        diffusion = torch.Tensor([[diffusion]]).to(self.device)
        for layer in self.layers():
            if isinstance(layer, LinearFlipout):
                layer.broaden(diffusion)
            if isinstance(layer, Conv2dFlipout):
                layer.broaden(diffusion)
        self.set_trainable(True)

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, ModelDNN)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = None
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        new_model.load_state_dict(statedict)
        return new_model

    def kl(self, st):
        kl = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
            elif isinstance(layer, LinearFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
        return kl, kl / self.weight_kl_dense, torch.tensor(0.).to(self.device)

    @classmethod
    def get_surrogate_prior(cls, path, model):
        assert False, 'not use'
        state_dict = torch.load(path)
        prior = [state_dict['conv1.mu_kernel'], state_dict['conv1.rho_kernel'],
                 state_dict['conv1.mu_bias'], state_dict['conv1.rho_bias'],
                 state_dict['conv2.mu_kernel'], state_dict['conv2.rho_kernel'],
                 state_dict['conv2.mu_bias'], state_dict['conv2.rho_bias'],
                 state_dict['conv3.mu_kernel'], state_dict['conv3.rho_kernel'],
                 state_dict['conv3.mu_bias'], state_dict['conv3.rho_bias'],
                 state_dict['conv4.mu_kernel'], state_dict['conv4.rho_kernel'],
                 state_dict['conv4.mu_bias'], state_dict['conv4.rho_bias'],
                 state_dict['dense.mu_weight'], state_dict['dense.rho_weight'],
                 state_dict['dense.mu_bias'], state_dict['dense.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class ModelPMNIST(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(ModelPMNIST, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 12
        else:
            prior = list(prior)

        if init_posterior is None:
            init_posterior = [None] * 12

        self.flatten = nn.Flatten().to(device)
        # self.dense1 = nn.Linear(28*28, 100).to(device)
        self.dense1 = LinearFlipout(
            28*28, 100,
            prior_mean=prior[0], prior_rho=prior[1],
            prior_mean_bias=prior[2], prior_rho_bias=prior[3],
            init_posterior_mean=init_posterior[0], init_posterior_rho=init_posterior[1],
            init_posterior_mean_bias=init_posterior[2], init_posterior_rho_bias=init_posterior[3]).to(device)
        self.relu1 = nn.ReLU().to(device)
        # self.dense2 = nn.Linear(100, 100).to(device)
        self.dense2 = LinearFlipout(
            100, 100,
            prior_mean=prior[4], prior_rho=prior[5],
            prior_mean_bias=prior[6], prior_rho_bias=prior[7],
            init_posterior_mean=init_posterior[4], init_posterior_rho=init_posterior[5],
            init_posterior_mean_bias=init_posterior[6], init_posterior_rho_bias=init_posterior[7]).to(device)
        self.relu2 = nn.ReLU().to(device)
        # self.dense3 = nn.Linear(100, 10).to(device)
        self.dense3 = LinearFlipout(
            100, 10,
            prior_mean=prior[8], prior_rho=prior[9],
            prior_mean_bias=prior[10], prior_rho_bias=prior[11],
            init_posterior_mean=init_posterior[8], init_posterior_rho=init_posterior[9],
            init_posterior_mean_bias=init_posterior[10], init_posterior_rho_bias=init_posterior[11]).to(device)

    def forward(self, X):
        output = X
        for layer in self.layers():
            output = layer(output)
        return output

    def layers(self):
        return [self.flatten,
                self.dense1,
                self.relu1,
                self.dense2,
                self.relu2,
                self.dense3]

    def broaden(self, diffusion):
        self.set_trainable(False)
        diffusion = torch.Tensor([[diffusion]]).to(self.device)
        for layer in self.layers():
            if isinstance(layer, LinearFlipout):
                layer.broaden(diffusion)
            if isinstance(layer, Conv2dFlipout):
                layer.broaden(diffusion)
        self.set_trainable(True)

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def pertube(self):
        assert False, 'not to use!'
        if self.config.droprate_init_strategy == 0:
            pass
        elif self.config.droprate_init_strategy == 1:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.rand_like(layer.droprate_factor)
                    layer.droprate_factor.data = layer.droprate_factor.data * eps
        elif self.config.droprate_init_strategy == 2:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.normal(mean=0.0, std=torch.Tensor(
                        [1.0]), device=layer.device)
                    layer.droprate_factor.data = layer.droprate_factor.data + eps

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, ModelPMNIST)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = [statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    def kl(self, st):
        kl = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
            elif isinstance(layer, LinearFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
        return kl, kl / self.weight_kl_dense, torch.tensor(0.).to(self.device)

    @classmethod
    def get_surrogate_prior(cls, path, model):
        statedict = torch.load(path)
        prior = [statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class DropoutModelPMNIST(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st / runlength
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(DropoutModelPMNIST, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 15

        if init_posterior is None:
            init_posterior = [None] * 15

        self.flatten = nn.Flatten().to(device)
        self.dropout1 = VariationalBernoulliDropout(
            in_features=28*28, inv_sigmoid_prior_droprate=prior[0],
            init_inv_sigmoid_posterior_droprate=init_posterior[0]).to(device)
        self.dense1 = LinearFlipout(
            in_features=28*28, out_features=100,
            prior_mean=prior[1], prior_rho=prior[2],
            prior_mean_bias=prior[3], prior_rho_bias=prior[4],
            init_posterior_mean=init_posterior[1], init_posterior_rho=init_posterior[2],
            init_posterior_mean_bias=init_posterior[3], init_posterior_rho_bias=init_posterior[4]).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.dropout2 = VariationalBernoulliDropout(
            in_features=100, inv_sigmoid_prior_droprate=prior[5],
            init_inv_sigmoid_posterior_droprate=init_posterior[5]).to(device)
        self.dense2 = LinearFlipout(
            in_features=100, out_features=100,
            prior_mean=prior[6], prior_rho=prior[7],
            prior_mean_bias=prior[8], prior_rho_bias=prior[9],
            init_posterior_mean=init_posterior[6], init_posterior_rho=init_posterior[7],
            init_posterior_mean_bias=init_posterior[8], init_posterior_rho_bias=init_posterior[9]).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.dropout3 = VariationalBernoulliDropout(
            in_features=100, inv_sigmoid_prior_droprate=prior[10],
            init_inv_sigmoid_posterior_droprate=init_posterior[10]).to(device)
        self.dense3 = LinearFlipout(
            in_features=100, out_features=10,
            prior_mean=prior[11], prior_rho=prior[12],
            prior_mean_bias=prior[13], prior_rho_bias=prior[14],
            init_posterior_mean=init_posterior[11], init_posterior_rho=init_posterior[12],
            init_posterior_mean_bias=init_posterior[13], init_posterior_rho_bias=init_posterior[14]).to(device)

    def forward(self, X, st=0):
        output = X
        for layer in self.layers():
            if isinstance(layer, VariationalBernoulliDropout):
                output = layer(output, st)
            else:
                output = layer(output)
        return output

    def layers(self):
        return [self.flatten,
                self.dropout1, self.dense1, self.relu1,
                self.dropout2, self.dense2, self.relu2,
                self.dropout3, self.dense3]

    def broaden(self, diffusion):
        pass

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def pertube(self):
        assert False, 'not to use!'
        if self.config.droprate_init_strategy == 0:
            pass
        elif self.config.droprate_init_strategy == 1:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.rand_like(layer.droprate_factor)
                    layer.droprate_factor.data = layer.droprate_factor.data * eps
        elif self.config.droprate_init_strategy == 2:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.normal(mean=0.0, std=torch.Tensor(
                        [1.0]), device=layer.device)
                    layer.droprate_factor.data = layer.droprate_factor.data + eps

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, DropoutModelPMNIST)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = [statedict['dropout1.inv_sigmoid_droprate'],
                 statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dropout2.inv_sigmoid_droprate'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dropout3.inv_sigmoid_droprate'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    @classmethod
    def kl_divergence(cls, Model_q, Model_p):
        assert False, 'not to use!'
        assert isinstance(Model_q, Model)
        assert isinstance(Model_p, Model)

        res = 0.0
        for (layer_q, layer_p) in zip(Model_q.layers(), Model_p.layers()):
            if isinstance(layer_q, BernoulliDropout):
                kl_dropout = BernoulliDropout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dropout * kl_dropout
            elif isinstance(layer_q, Dense):
                kl_dense = Dense.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, LinearFlipout):
                kl_dense = LinearFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, Conv2dFlipout):
                kl_dense = Conv2dFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
        return res

    def kl(self, st):
        kl = torch.tensor(0., device=self.device)
        kl_gauss = torch.tensor(0., device=self.device)
        kl_dropout = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, LinearFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, VariationalBernoulliDropout):
                kl_tmp = layer.kl_loss(st)
                kl += kl_tmp * self.weight_kl_dropout
                kl_dropout += kl_tmp
        return kl, kl_gauss, kl_dropout

    @classmethod
    def get_surrogate_prior(cls, path, model):
        statedict = torch.load(path)
        prior = [statedict['dropout1.inv_sigmoid_droprate'],
                 statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dropout2.inv_sigmoid_droprate'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dropout3.inv_sigmoid_droprate'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class NonAdaptiveDropoutModelPMNIST(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st / runlength
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(NonAdaptiveDropoutModelPMNIST, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 15

        if init_posterior is None:
            init_posterior = [None] * 15

        self.flatten = nn.Flatten().to(device)
        self.dropout1 = nn.Dropout(p=0.2).to(device)
        self.dense1 = LinearFlipout(
            in_features=28*28, out_features=100,
            prior_mean=prior[1], prior_rho=prior[2],
            prior_mean_bias=prior[3], prior_rho_bias=prior[4],
            init_posterior_mean=init_posterior[1], init_posterior_rho=init_posterior[2],
            init_posterior_mean_bias=init_posterior[3], init_posterior_rho_bias=init_posterior[4]).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.dropout2 = nn.Dropout(p=0.2).to(device)
        self.dense2 = LinearFlipout(
            in_features=100, out_features=100,
            prior_mean=prior[6], prior_rho=prior[7],
            prior_mean_bias=prior[8], prior_rho_bias=prior[9],
            init_posterior_mean=init_posterior[6], init_posterior_rho=init_posterior[7],
            init_posterior_mean_bias=init_posterior[8], init_posterior_rho_bias=init_posterior[9]).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.dropout3 = nn.Dropout(p=0.2).to(device)
        self.dense3 = LinearFlipout(
            in_features=100, out_features=10,
            prior_mean=prior[11], prior_rho=prior[12],
            prior_mean_bias=prior[13], prior_rho_bias=prior[14],
            init_posterior_mean=init_posterior[11], init_posterior_rho=init_posterior[12],
            init_posterior_mean_bias=init_posterior[13], init_posterior_rho_bias=init_posterior[14]).to(device)

    def forward(self, X, st=0):
        output = X
        for layer in self.layers():
            if isinstance(layer, VariationalBernoulliDropout):
                output = layer(output, st)
            else:
                output = layer(output)
        return output

    def layers(self):
        return [self.flatten,
                self.dropout1, self.dense1, self.relu1,
                self.dropout2, self.dense2, self.relu2,
                self.dropout3, self.dense3]

    def broaden(self, diffusion):
        pass

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def pertube(self):
        assert False, 'not to use!'
        if self.config.droprate_init_strategy == 0:
            pass
        elif self.config.droprate_init_strategy == 1:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.rand_like(layer.droprate_factor)
                    layer.droprate_factor.data = layer.droprate_factor.data * eps
        elif self.config.droprate_init_strategy == 2:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.normal(mean=0.0, std=torch.Tensor(
                        [1.0]), device=layer.device)
                    layer.droprate_factor.data = layer.droprate_factor.data + eps

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, NonAdaptiveDropoutModelPMNIST)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = [None,
                 statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 None,
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 None,
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    @classmethod
    def kl_divergence(cls, Model_q, Model_p):
        assert False, 'not to use!'
        assert isinstance(Model_q, Model)
        assert isinstance(Model_p, Model)

        res = 0.0
        for (layer_q, layer_p) in zip(Model_q.layers(), Model_p.layers()):
            if isinstance(layer_q, BernoulliDropout):
                kl_dropout = BernoulliDropout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dropout * kl_dropout
            elif isinstance(layer_q, Dense):
                kl_dense = Dense.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, LinearFlipout):
                kl_dense = LinearFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, Conv2dFlipout):
                kl_dense = Conv2dFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
        return res

    def kl(self, st):
        kl = torch.tensor(0., device=self.device)
        kl_gauss = torch.tensor(0., device=self.device)
        kl_dropout = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, LinearFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, VariationalBernoulliDropout):
                kl_tmp = layer.kl_loss(st)
                kl += kl_tmp * self.weight_kl_dropout
                kl_dropout += kl_tmp
        return kl, kl_gauss, kl_dropout

    @classmethod
    def get_surrogate_prior(cls, path, model):
        statedict = torch.load(path)
        prior = [statedict['dropout1.inv_sigmoid_droprate'],
                 statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dropout2.inv_sigmoid_droprate'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dropout3.inv_sigmoid_droprate'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class GaussianDropoutModelPMNIST(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st / runlength
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(GaussianDropoutModelPMNIST, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 15

        if init_posterior is None:
            init_posterior = [None] * 15

        self.flatten = nn.Flatten().to(device)
        self.dropout1 = GaussDropout(
            tasks=1, input_size=28*28).to(device)
        self.dense1 = LinearFlipout(
            in_features=28*28, out_features=100,
            prior_mean=prior[1], prior_rho=prior[2],
            prior_mean_bias=prior[3], prior_rho_bias=prior[4],
            init_posterior_mean=init_posterior[1], init_posterior_rho=init_posterior[2],
            init_posterior_mean_bias=init_posterior[3], init_posterior_rho_bias=init_posterior[4]).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.dropout2 = GaussDropout(
            tasks=1, input_size=100).to(device)
        self.dense2 = LinearFlipout(
            in_features=100, out_features=100,
            prior_mean=prior[6], prior_rho=prior[7],
            prior_mean_bias=prior[8], prior_rho_bias=prior[9],
            init_posterior_mean=init_posterior[6], init_posterior_rho=init_posterior[7],
            init_posterior_mean_bias=init_posterior[8], init_posterior_rho_bias=init_posterior[9]).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.dropout3 = GaussDropout(
            tasks=1, input_size=100).to(device)
        self.dense3 = LinearFlipout(
            in_features=100, out_features=10, prior_mean=prior[11], prior_rho=prior[12],
            prior_mean_bias=prior[13], prior_rho_bias=prior[14],
            init_posterior_mean=init_posterior[11], init_posterior_rho=init_posterior[12],
            init_posterior_mean_bias=init_posterior[13], init_posterior_rho_bias=init_posterior[14]).to(device)
        
    def forward(self, X, st=0):
        output = X
        for layer in self.layers():
            if isinstance(layer, VariationalBernoulliDropout):
                output = layer(output, st)
            elif isinstance(layer, GaussDropout) and st == 0:
                output = output
            elif isinstance(layer, GaussDropout) and st == 1:
                output = layer(output, torch.tensor(0, device=self.device))
            else:
                output = layer(output)
        return output

    def layers(self):
        return [self.flatten,
                self.dropout1, self.dense1, self.relu1,
                self.dropout2, self.dense2, self.relu2,
                self.dropout3, self.dense3]

    def broaden(self, diffusion):
        pass

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def pertube(self):
        assert False, 'not to use!'
        if self.config.droprate_init_strategy == 0:
            pass
        elif self.config.droprate_init_strategy == 1:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.rand_like(layer.droprate_factor)
                    layer.droprate_factor.data = layer.droprate_factor.data * eps
        elif self.config.droprate_init_strategy == 2:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.normal(mean=0.0, std=torch.Tensor(
                        [1.0]), device=layer.device)
                    layer.droprate_factor.data = layer.droprate_factor.data + eps

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, GaussianDropoutModelPMNIST)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = [None,
                 statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 None,
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 None,
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    @classmethod
    def kl_divergence(cls, Model_q, Model_p):
        assert False, 'not to use!'
        assert isinstance(Model_q, Model)
        assert isinstance(Model_p, Model)

        res = 0.0
        for (layer_q, layer_p) in zip(Model_q.layers(), Model_p.layers()):
            if isinstance(layer_q, BernoulliDropout):
                kl_dropout = BernoulliDropout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dropout * kl_dropout
            elif isinstance(layer_q, Dense):
                kl_dense = Dense.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, LinearFlipout):
                kl_dense = LinearFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, Conv2dFlipout):
                kl_dense = Conv2dFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
        return res

    def kl(self):
        kl = torch.tensor(0., device=self.device)
        kl_gauss = torch.tensor(0., device=self.device)
        kl_dropout = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, LinearFlipout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dense
                kl_gauss += kl_tmp
            elif isinstance(layer, VariationalBernoulliDropout):
                kl_tmp = layer.kl_loss()
                kl += kl_tmp * self.weight_kl_dropout
                kl_dropout += kl_tmp
            elif isinstance(layer, GaussDropout):
                kl_tmp = layer.kl_loss(torch.tensor(0, device=self.device))
                kl += kl_tmp * self.weight_kl_dropout
                kl_dropout += kl_tmp
        return kl, kl_gauss, kl_dropout

    @classmethod
    def get_surrogate_prior(cls, path, model):
        statedict = torch.load(path)
        prior = [None,
                 statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 None,
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 None,
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class ModelDNNPMNIST(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(ModelDNNPMNIST, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 12
        else:
            prior = list(prior)

        if init_posterior is None:
            init_posterior = [None] * 12

        self.flatten = nn.Flatten().to(device)
        # self.dense1 = nn.Linear(28*28, 100).to(device)
        self.dense1 = nn.Linear(28*28, 100).to(device)
        self.relu1 = nn.ReLU().to(device)
        # self.dense2 = nn.Linear(100, 100).to(device)
        self.dense2 = nn.Linear(100, 100).to(device)
        self.relu2 = nn.ReLU().to(device)
        # self.dense3 = nn.Linear(100, 10).to(device)
        self.dense3 = nn.Linear(100, 10).to(device)
        

    def forward(self, X):
        output = X
        for layer in self.layers():
            output = layer(output)
        return output

    def layers(self):
        return [self.flatten,
                self.dense1,
                self.relu1,
                self.dense2,
                self.relu2,
                self.dense3]

    def broaden(self, diffusion):
        self.set_trainable(False)
        diffusion = torch.Tensor([[diffusion]]).to(self.device)
        for layer in self.layers():
            if isinstance(layer, LinearFlipout):
                layer.broaden(diffusion)
            if isinstance(layer, Conv2dFlipout):
                layer.broaden(diffusion)
        self.set_trainable(True)

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def pertube(self):
        assert False, 'not to use!'
        if self.config.droprate_init_strategy == 0:
            pass
        elif self.config.droprate_init_strategy == 1:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.rand_like(layer.droprate_factor)
                    layer.droprate_factor.data = layer.droprate_factor.data * eps
        elif self.config.droprate_init_strategy == 2:
            for layer in self.layers():
                if isinstance(layer, BernoulliDropout):
                    eps = torch.normal(mean=0.0, std=torch.Tensor(
                        [1.0]), device=layer.device)
                    layer.droprate_factor.data = layer.droprate_factor.data + eps

    def print(self):
        logger = logging.getLogger('logger')
        for name, param in self.named_parameters():
            logger.info(name, param.data)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, ModelDNNPMNIST)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = None
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        new_model.load_state_dict(statedict)
        return new_model

    @classmethod
    def kl_divergence(cls, Model_q, Model_p):
        assert False, 'not to use!'
        assert isinstance(Model_q, Model)
        assert isinstance(Model_p, Model)

        res = torch.tensor(0., device=self.device)
        for (layer_q, layer_p) in zip(Model_q.layers(), Model_p.layers()):
            if isinstance(layer_q, BernoulliDropout):
                kl_dropout = BernoulliDropout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dropout * kl_dropout
            elif isinstance(layer_q, Dense):
                kl_dense = Dense.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, LinearFlipout):
                kl_dense = LinearFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
            elif isinstance(layer_q, Conv2dFlipout):
                kl_dense = Conv2dFlipout.kl_divergence(layer_q, layer_p)
                res += Model_q.weight_kl_dense * kl_dense
        return res

    def kl(self):
        kl = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
            elif isinstance(layer, LinearFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
        return kl, kl / self.weight_kl_dense, torch.tensor(0.).to(self.device)

    @classmethod
    def get_surrogate_prior(cls, path, model):
        statedict = torch.load(path)
        prior = [statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model


class ModelToy(nn.Module):
    def __init__(self, config, st, hidden, prior=None, init_posterior=None, device='cpu'):
        '''
        :param config: config object
        :param st: st
        :param hidden: hidden layers
        :param prior: prior: list of tensors
        :param device: device
        '''
        super(ModelPMNIST, self).__init__()
        self.config = config
        self.device = device
        self.st = st
        self.hidden = hidden  # not used
        self.weight_kl_dropout = config.weight_kl_dropout
        self.weight_kl_dense = config.weight_kl_dense
        self.conv_kl = torch.zeros(size=(1,), device=device)

        if prior is None:
            prior = [None] * 12
        else:
            prior = list(prior)

        if init_posterior is None:
            init_posterior = [None] * 12

        self.flatten = nn.Flatten().to(device)
        self.dense1 = LinearFlipout(
            2, 10,
            prior_mean=prior[0], prior_rho=prior[1],
            prior_mean_bias=prior[2], prior_rho_bias=prior[3],
            init_posterior_mean=init_posterior[0], init_posterior_rho=init_posterior[1],
            init_posterior_mean_bias=init_posterior[2], init_posterior_rho_bias=init_posterior[3]).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.dense2 = LinearFlipout(
            10, 1,
            prior_mean=prior[4], prior_rho=prior[5],
            prior_mean_bias=prior[6], prior_rho_bias=prior[7],
            init_posterior_mean=init_posterior[4], init_posterior_rho=init_posterior[5],
            init_posterior_mean_bias=init_posterior[6], init_posterior_rho_bias=init_posterior[7]).to(device)
        self.relu2 = nn.ReLU().to(device)

    def forward(self, X):
        output = X
        for layer in self.layers():
            output = layer(output)
        return output

    def layers(self):
        return [self.flatten,
                self.dense1,
                self.relu1,
                self.dense2,
                self.relu2]

    def broaden(self, diffusion):
        self.set_trainable(False)
        diffusion = torch.Tensor([[diffusion]]).to(self.device)
        for layer in self.layers():
            if isinstance(layer, LinearFlipout):
                layer.broaden(diffusion)
            if isinstance(layer, Conv2dFlipout):
                layer.broaden(diffusion)
        self.set_trainable(True)

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    @classmethod
    def clone(cls, init_model):
        """
            Create a new instance of Model which have same params as init_model.
        """
        assert isinstance(init_model, ModelPMNIST)
        init_model.set_trainable(False)
        new_model = copy.deepcopy(init_model)
        new_model.set_trainable(True)
        return new_model

    @classmethod
    def from_posterior(cls, model):
        statedict = model.state_dict()
        prior = [statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

    def kl(self):
        kl = torch.tensor(0., device=self.device)
        for layer in self.layers():
            if isinstance(layer, Conv2dFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
            elif isinstance(layer, LinearFlipout):
                kl += layer.kl_loss() * self.weight_kl_dense
        return kl, kl / self.weight_kl_dense, torch.tensor(0.).to(self.device)

    @classmethod
    def get_surrogate_prior(cls, path, model):
        assert False, 'no use surrogate prior for this task!'
        statedict = torch.load(path)
        prior = [statedict['dense1.mu_weight'], statedict['dense1.rho_weight'],
                 statedict['dense1.mu_bias'], statedict['dense1.rho_bias'],
                 statedict['dense2.mu_weight'], statedict['dense2.rho_weight'],
                 statedict['dense2.mu_bias'], statedict['dense2.rho_bias'],
                 statedict['dense3.mu_weight'], statedict['dense3.rho_weight'],
                 statedict['dense3.mu_bias'], statedict['dense3.rho_bias']]
        init_posterior = prior
        new_model = cls(model.config, model.st, model.hidden,
                        prior, init_posterior, model.device)
        return new_model

if __name__ == "__main__":
    model_1 = Model(0, in_size=2, device='cpu')
    model_1.print()
