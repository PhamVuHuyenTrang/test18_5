from argparse import Namespace
import torch
import torch.nn as nn
import logging
import multiprocessing
from tqdm import tqdm
from BNN.Model import ModelPMNIST as Model
from BNN import utils
from copy import deepcopy
import time
import numpy as np
from multiprocessing.dummy import Pool
import torch.functional as F
import mnets
import hnets
import probabilistic
from probabilistic.gauss_hnet_init import gauss_hyperfan_init


class Hypothesis:
    def __init__(self, config: Namespace, st, device):
        self.config = config
        self.device = device
        self.st = st
        self.history = ""
        self.prior_prob = torch.Tensor([1.]).to(self.device)
        self.prior_prob.requires_grad_(False)
        self.prob = torch.Tensor([1.]).to(self.device)
        self.prob.requires_grad_(False)
        self.neg_celbo = torch.Tensor([1.]).to(self.device)

        self.prior_mnet_weight = None
        self.task_embedding = nn.Parameter(
            torch.randn((1, 64,)), requires_grad=True).detach()/ \
            torch.sqrt(torch.tensor(10.))
        self.task_embedding = self.task_embedding.to(self.device)

    def broaden(self, diffusion):
        for i in range(len(self.prior_mnet_weight)//2, len(self.prior_mnet_weight)):
            sigma = nn.functional.softplus(self.prior_mnet_weight[i])
            self.prior_mnet_weight[i] = utils.inv_softplus(diffusion*sigma)

    @classmethod
    def clone(cls, hypo, st, hnet):
        new_hypo = Hypothesis(hypo.config, st, hypo.device)
        new_hypo.history = hypo.history
        new_hypo.st = st

        new_hypo.prior_prob = hypo.prob.detach().clone()
        new_hypo.prior_prob = new_hypo.prior_prob.requires_grad_(False)

        new_hypo.prob = hypo.prob.detach().clone()
        new_hypo.prob = new_hypo.prob.requires_grad_(False)

        new_hypo.task_embedding = hypo.task_embedding.detach().clone()
        new_hypo.task_embedding = new_hypo.task_embedding.requires_grad_(True)

        weight = hnet.forward(cond_input=hypo.task_embedding)
        new_hypo.prior_mnet_weight = [None] * len(weight)
        for i in range(len(new_hypo.prior_mnet_weight)):
            new_hypo.prior_mnet_weight[i] = weight[i].detach().clone()
            new_hypo.prior_mnet_weight[i] = new_hypo.prior_mnet_weight[i].requires_grad_(
                False)

        if st == 1:
            new_hypo.broaden(hypo.config.diffusion)
        return new_hypo


class HypothesesStorage():
    def __init__(self, config: Namespace, device):
        self.device = device
        self.config = config
        self.hypotheses = []
        self.num_models = self.config.num_models  # for pruning
        self.Lambda = torch.Tensor([self.config.Lambda]).to(device)
        self.test_abserr = []
        self.batch_size = self.config.batch_size
        self.loss = lambda x, y: (x-y)**2/(2*config.noise**2)
        self.mnet = probabilistic.GaussianMLPFlipout(129, 1, [100, 100]).to(device)
        self.mnet = probabilistic.gauss_mnet_interface.GaussianBNNWrapper(
            self.mnet, apply_rho_offset=True)
        self.hnet = hnets.chunked_mlp_hnet.ChunkedHMLP(
            self.mnet.param_shapes, uncond_in_size=0, cond_in_size=64,
            layers=[100, 100], chunk_size=128, use_batch_norm=True).to(device)
        # gauss_hyperfan_init(self.hnet, self.mnet)

    def get_hnet_and_embedding_from_pretrain(self, hnet_path, emb_path):
        state_dict = torch.load(hnet_path)
        self.hnet.load_state_dict(state_dict)
        self.hypotheses[0].task_embedding = torch.load(emb_path)

    def __add_and_cal_prior_prob_vbs(self):
        if len(self.hypotheses) >= 1:
            new_hypotheses = []
            for hypo in self.hypotheses:
                hypo_s0 = Hypothesis.clone(hypo, st=0, hnet=self.hnet)
                hypo_s0.history += "0"
                hypo_s0.prior_prob = hypo.prob * (1. - self.Lambda)
                hypo_s0.prior_prob = hypo_s0.prior_prob.detach().requires_grad_(False)
                new_hypotheses += [hypo_s0]
                # hypo_s0.prior_prob = torch.ones_like(hypo_s0.prior_prob).detach().requires_grad_(False)
                if not self.config.train_from_scratch:
                    hypo_s1 = Hypothesis.clone(hypo, st=1, hnet=self.hnet)
                    hypo_s1.history += "1"
                    hypo_s1.prior_prob = hypo.prob * self.Lambda
                    hypo_s1.prior_prob = hypo_s1.prior_prob.detach().requires_grad_(False)
                    new_hypotheses += [hypo_s1]
            self.hypotheses = new_hypotheses
        else:
            hypo_s1 = Hypothesis(config=self.config, st=1, device=self.device)
            hypo_s1.history += "1"
            hypo_s1.prior_prob = torch.Tensor([1.]).to(
                self.device).detach().requires_grad_(False)
            hypo_s1.prior_mnet_weight = self.hnet.forward(
                cond_input=hypo_s1.task_embedding)
            for i in range(len(hypo_s1.prior_mnet_weight)):
                hypo_s1.prior_mnet_weight[i] = hypo_s1.prior_mnet_weight[i].detach(
                )
                hypo_s1.prior_mnet_weight[i] = hypo_s1.prior_mnet_weight[i].requires_grad_(
                    False)
            self.hypotheses = [hypo_s1]
            if not self.config.train_from_scratch:
                self.get_hnet_and_embedding_from_pretrain(
                    self.config.hnet_path, self.config.emb_path)

    def __normalize_prob(self):
        if len(self.hypotheses) > 1:
            sum = torch.zeros(1).to(self.device)
            for hypo in self.hypotheses:
                sum += hypo.prob
            for hypo in self.hypotheses:
                hypo.prob /= sum
        else:
            self.hypotheses[0].prob = torch.Tensor(
                [1.]).to(self.device).detach()

    def prune_vbs(self):
        if len(self.hypotheses) > self.num_models:
            self.hypotheses = sorted(self.hypotheses,
                                     reverse=True,
                                     key=lambda hypo: hypo.prob)
            self.hypotheses = self.hypotheses[:self.num_models]
        self.__normalize_prob()

    def cal_ncelbo(self, X, y, embedding, prior_mnet_weight, finalround=False):
        # only for mcSamples = 1
        if finalround:
            X_tiled = torch.tile(X, (self.config.mcSamples_finalround,) +
                                 (1,)*(len(X.shape)-1))
            y_tiled = torch.tile(y, (self.config.mcSamples_finalround,) +
                                 (1,)*(len(y.shape)-1))
        else:
            X_tiled = torch.tile(X, (self.config.mcSamples,) +
                                 (1,)*(len(X.shape)-1))
            y_tiled = torch.tile(y, (self.config.mcSamples,) +
                                 (1,)*(len(y.shape)-1))
        mnet_weight = self.hnet.forward(cond_input=embedding)
        X_tiled = nn.Flatten()(X_tiled)
        y_pred = self.mnet.forward(X_tiled, weights=mnet_weight)
        nll = torch.mean(self.loss(y_pred, y_tiled))
        w_mean, w_rho = self.mnet.extract_mean_and_rho(weights=mnet_weight)
        _, w_logvar = utils.decode_diag_gauss(
            w_rho, logvar_enc=self.mnet.logvar_encoding, return_logvar=True)
        prior_mean, prior_rho = self.mnet.extract_mean_and_rho(
            weights=prior_mnet_weight)
        _, prior_logvar = utils.decode_diag_gauss(
            prior_rho, logvar_enc=self.mnet.logvar_encoding, return_logvar=True)
        kl = utils.kl_diag_gaussians(w_mean, w_logvar,
                                     prior_mean, prior_logvar)*self.config.weight_kl_dense
        return nll + kl/X.shape[0], nll, kl/X.shape[0]

    def update_vbs(self, X, y, epochs):
        logger = logging.getLogger('logger')
        self.__add_and_cal_prior_prob_vbs()
        emb_list = [hypo.task_embedding for hypo in self.hypotheses]
        optimizer = torch.optim.Adam(list(self.hnet.parameters())
                                     + emb_list,
                                     lr=self.config.learning_rate)

        for epoch_id in tqdm(range(epochs)):
            optimizer.zero_grad()
            nelbo = torch.zeros((1)).to(self.device)
            for hypo in self.hypotheses:
                hypo.neg_celbo, nll, kl = self.cal_ncelbo(
                    X, y, hypo.task_embedding, hypo.prior_mnet_weight)
                #print(hypo.neg_celbo)
                hypo.neg_celbo = hypo.neg_celbo.reshape((1))
                if hypo.neg_celbo > 10.:
                    hypo.prob = hypo.prior_prob / hypo.neg_celbo
                else:
                    hypo.prob = hypo.prior_prob * torch.exp(-hypo.neg_celbo)
                logger.info(
                    f'epoch: {epoch_id} --- task: {hypo.history} --- nll: {nll.item()} --- kl: {kl.item()} --- prob: {hypo.prob.item()}')
            self.__normalize_prob()

            for hypo in self.hypotheses:
                if hypo.prob < 1e-2:  # for computational stability
                    continue
                nelbo += hypo.prob * (hypo.neg_celbo +
                                      torch.log(hypo.prob/hypo.prior_prob))
            logger.info(f'epoch: {epoch_id} --- nelbo: {nelbo.item()}')
            nelbo.backward()
            optimizer.step()

        for hypo in self.hypotheses:
            prior_mnet_weight = self.hnet.forward(
                cond_input=hypo.task_embedding)
            hypo.neg_celbo, _, _ = self.cal_ncelbo(
                X, y, hypo.task_embedding, prior_mnet_weight, finalround=True)
            hypo.neg_celbo = hypo.neg_celbo.reshape((1))
            if hypo.neg_celbo > 100.:
                hypo.prob = hypo.prior_prob / hypo.neg_celbo
            else:
                hypo.prob = hypo.prior_prob * torch.exp(-hypo.neg_celbo)
        self.__normalize_prob()

        if self.config.prune:
            self.prune_vbs()

        # update after prune
        for epoch_id in tqdm(range(epochs)):
            optimizer.zero_grad()
            nelbo = torch.zeros((1)).to(self.device)
            for hypo in self.hypotheses:
                hypo.neg_celbo, nll, kl = self.cal_ncelbo(
                    X, y, hypo.task_embedding, hypo.prior_mnet_weight)
                hypo.neg_celbo = hypo.neg_celbo.reshape((1))
                if hypo.neg_celbo > 10.:
                    hypo.prob = hypo.prior_prob / hypo.neg_celbo / 1000.
                else:
                    hypo.prob = hypo.prior_prob * torch.exp(-hypo.neg_celbo)
                logger.info(
                    f'epoch: {epoch_id} --- task: {hypo.history} --- nll: {nll.item()} --- kl: {kl.item()} --- prob: {hypo.prob.item()}')
            self.__normalize_prob()

            for hypo in self.hypotheses:
                if hypo.prob < 1e-2:  # for computational stability
                    continue
                nelbo += hypo.prob * (hypo.neg_celbo +
                                      torch.log(hypo.prob/hypo.prior_prob))
            logger.info(f'epoch: {epoch_id} --- nelbo: {nelbo.item()}')
            nelbo.backward()
            optimizer.step()

        for hypo in self.hypotheses:
            prior_mnet_weight = self.hnet.forward(
                cond_input=hypo.task_embedding)
            hypo.neg_celbo, _, _ = self.cal_ncelbo(
                X, y, hypo.task_embedding, prior_mnet_weight, finalround=True)
            hypo.neg_celbo = hypo.neg_celbo.reshape((1))
            if hypo.neg_celbo > 100.:
                hypo.prob = hypo.prior_prob / hypo.neg_celbo
            else:
                hypo.prob = hypo.prior_prob * torch.exp(-hypo.neg_celbo)
        self.__normalize_prob()

        self.log_vbs()

    def log_vbs(self):
        logger = logging.getLogger("logger")
        for hypo in self.hypotheses:
            logger.info(
                f'history: {hypo.history} --- prob: {hypo.prob.item()} --- ncelbo: {hypo.neg_celbo.item()}')

    def test_ensemble(self, X_test, y_test):
        n_correct = torch.tensor(0, device=self.device)

        y_pred = torch.zeros((y_test.shape[0], 10)).to(self.device)
        for hypo in self.hypotheses:
            prior_mnet_weight = self.hnet.forward(
                cond_input=hypo.task_embedding)
            X_test = nn.Flatten()(X_test)
            y_pred += self.mnet.forward(X_test,
                                        weights=prior_mnet_weight) * hypo.prob
        y_pred = torch.argmax(y_pred, dim=1)
        # y_test_id = y_test.argmax(dim=-1)
        n_correct += torch.sum(y_pred == y_test)
        accuracy = n_correct/y_test.shape[0]
        logger = logging.getLogger("logger")
        logger.info(f'acc: {accuracy}')
        return accuracy
    
    def calculate_abserr(self, X_test, y_test, log_odds):
        prior_mnet_weight = self.hnet.forward(cond_input = self.hypotheses[0].task_embedding)
        y_pred = self.mnet.forward(X_test, weights = prior_mnet_weight)
        if log_odds:
            y_pred = torch.sigmoid(y_pred)
        return torch.abs(y_pred-y_test)
    def test_real_data(self, X_test, y_test, log_odds):

        abserr = torch.mean(self.calculate_abserr(X_test, y_test, log_odds))
        self.test_abserr.append(abserr)


def update_qz_job(a):
    '''
        job for multiprocessing
    '''
    hypothesis = a[0]
    X = a[1]
    y = a[2]
    hypothesis.cal_qz(X, y)
    return hypothesis
