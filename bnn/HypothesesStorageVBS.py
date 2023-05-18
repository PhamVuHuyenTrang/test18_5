from argparse import Namespace
import torch
import torch.nn as nn
import logging
import multiprocessing
from tqdm import tqdm
from BNN.Model import DropoutModel as Model
from copy import deepcopy
import time
import numpy as np
from multiprocessing.dummy import Pool
import mnets
import hnets
import probabilistic


class Hypothesis:
    def __init__(self, config: Namespace, st, device, mnet, hnet):
        self.config = config
        self.device = device
        self.st = st
        self.history = ""
        self.prob = torch.Tensor([self.config.Lambda]).to(self.device)
        self.logprob = torch.Tensor([0.]).to(device)  # used for vbs
        self.prob.requires_grad_(False)
        self.neg_celbo = None
        self.batch_size = self.config.batch_size

        self.model = Model(self.config, self.st,
                           hidden=[1], device=self.device)
        # self.model_embedding = model_embedding
        self.mnet = None  # not used
        self.hnet = None  # not used
        self.delta_hnet = None  # not used
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def cal_qz(self, X, y):
        logger = logging.getLogger('logger')
        logger.info(f'history: {self.history}')
        optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                     lr=self.config.learning_rate)
        epochs = self.config.epochs
        self.model.set_trainable(True)
        self.model.train()
        min_neg_celbo = torch.inf
        id = 0
        for epoch_id in tqdm(range(epochs + 1)):
            epoch_loss = 0.
            epoch_nll = 0.
            epoch_kl = 0.
            epoch_kl_gauss = 0.
            epoch_kl_dropout = 0.
            epoch_samples = 0
            for batch_id in range(0, X.shape[0], self.batch_size):
                optimizer.zero_grad()
                X_batch = X[batch_id:batch_id+self.batch_size]
                y_batch = y[batch_id:batch_id+self.batch_size]
                neg_celbo, nll, kl, kl_gauss, kl_dropout = self.__cal_neg_celbo(
                    X_batch, y_batch, self.model, None)
                if epoch_id > 0:
                    neg_celbo.backward()
                    optimizer.step()
                epoch_loss += neg_celbo.item() * X_batch.shape[0]
                epoch_nll += nll.item() * X_batch.shape[0]
                epoch_kl += kl.item() * X_batch.shape[0]
                epoch_kl_gauss += kl_gauss.item() * X_batch.shape[0]
                epoch_kl_dropout += kl_dropout.item() * X_batch.shape[0]
                epoch_samples += X_batch.shape[0]
            if epoch_loss < min_neg_celbo:
                min_neg_celbo = epoch_loss
                id = epoch_id
            if epoch_id > 200 and epoch_id > id + 50:
                print(f'Early stopping at epoch: {epoch_id}')
                break
            logger.info('Epoch: {}, loss: {}, nll: {}, kl: {}, kl_gauss: {}, kl_dropout: {}'.format(
                epoch_id, epoch_loss / epoch_samples, epoch_nll / epoch_samples,
                epoch_kl / epoch_samples, epoch_kl_gauss / epoch_samples,
                epoch_kl_dropout / epoch_samples))

        self.model.set_trainable(False)
        self.model.eval()
        self.neg_celbo = torch.tensor(0.0, device=self.device)
        for batch_id in range(0, X.shape[0], self.batch_size):
            X_batch = X[batch_id:batch_id+self.batch_size]
            y_batch = y[batch_id:batch_id+self.batch_size]
            ncelbo, _, _, _, _ = self.__cal_neg_celbo(
                X_batch, y_batch, self.model, None, finalround=True)
            self.neg_celbo += ncelbo * X_batch.shape[0]
        self.neg_celbo /= X.shape[0]
        logger.info(f'final: {self.neg_celbo}')

    def __cal_neg_celbo(self, X, y, tmp_model: Model,
                        mnet_weights, finalround=False):
        if not finalround:
            X_tiled = torch.tile(X, (self.config.mcSamples,) +
                                    (1,)*(len(X.shape)-1))
            y_tiled = torch.tile(y, (self.config.mcSamples,) +
                                    (1,)*(len(y.shape)-1))
        else:
            X_tiled = torch.tile(X, (self.config.mcSamples_finalround,) +
                                    (1,)*(len(X.shape)-1))
            y_tiled = torch.tile(y, (self.config.mcSamples_finalround,) +
                                    (1,)*(len(y.shape)-1))

        if self.config.variational_dropout:
            # forward in case use variational dropout
            z = tmp_model(X_tiled, self.st)
        else:
            # forward in case not use variational dropout
            z = tmp_model(X_tiled)

        neg_log_likelihood = self.loss(z, y_tiled)
        KL, kl_gauss, kl_dropout = tmp_model.kl(st=self.st)
        KL /= X.shape[0]
        kl_gauss /= X.shape[0]
        kl_dropout /= X.shape[0]
        neg_celbo = neg_log_likelihood + KL
        return neg_celbo, neg_log_likelihood, KL, kl_gauss, kl_dropout

    # def test_abserr(self, X_test, y_test, log_odds):  # for real_data
    #     list_y_pred = [self.model(X_test)
    #                    for _ in range(self.config.mcSamples_test)]
    #     if log_odds:
    #         list_y_pred = [torch.sigmoid(y_pred)
    #                        for y_pred in list_y_pred]  # log-odds
    #     y_pred = sum(list_y_pred) / len(list_y_pred)
    #     return torch.abs(y_pred-y_test).mean()

    @classmethod
    def clone(cls, hypo, st):
        new_hypo = Hypothesis(hypo.config, st, hypo.device,
                              hypo.mnet, hypo.hnet)
        new_hypo.history = hypo.history
        new_hypo.loss = hypo.loss
        new_hypo.st = st
        new_hypo.model = Model.from_posterior(hypo.model)
        new_hypo.neg_celbo = None
        new_hypo.prob = hypo.prob
        new_hypo.logprob = hypo.logprob
        return new_hypo


class HypothesesStorage():
    def __init__(self, config: Namespace, device, mnet, hnet):
        self.device = device
        self.config = config
        self.hypotheses = []
        self.num_models = self.config.num_models  # for pruning
        self.Lambda = self.config.Lambda
        self.test_abserr = []  # for real data
        self.mnet = mnet
        self.hnet = hnet
        self.batch_size = self.config.batch_size
        # self.mnet = probabilistic.gauss_mnet_interface.GaussianBNNWrapper(
        #     self.mnet)
        # self.hnet = hnets.HMLP(self.mnet.param_shapes, cond_in_size=8,
        #                        layers=[1, 1]).to(device)
        # self.hnet.apply_hyperfan_init()

    def __add_and_cal_prior_prob_vbs(self):
        if len(self.hypotheses) >= 1:
            new_hypotheses = []
            for hypo in self.hypotheses:
                if not self.config.variational_dropout or not self.config.save_model:
                    hypo_s0 = Hypothesis.clone(hypo, st=0)
                    print(hypo_s0.st)
                    hypo_s0.history += "0"
                    new_hypotheses += [hypo_s0]
                if not self.config.save_model or self.config.variational_dropout:
                    hypo_s1 = Hypothesis.clone(hypo, st=1)
                    print(hypo_s1.st)
                    hypo_s1.history += "1"
                    # if not self.config.train_from_scratch:
                    #     hypo_s1.model = Model.get_surrogate_prior(
                    #         self.config.surrogate_prior_path, hypo_s1.model).to(self.device)
                    if not self.config.variational_dropout:
                        hypo_s1.model.broaden(self.config.diffusion)
                    new_hypotheses += [hypo_s1]
            self.hypotheses = new_hypotheses
        else:
            hypo_s1 = Hypothesis(config=self.config, st=1, device=self.device,
                                 mnet=self.mnet, hnet=self.hnet)
            if not self.config.train_from_scratch:
                hypo_s1.model = Model.get_surrogate_prior(
                    self.config.surrogate_prior_path, hypo_s1.model).to(self.device)
            hypo_s1.history += "1"
            hypo_s1.model.broaden(self.config.diffusion)
            self.hypotheses = [hypo_s1]

    def __update_logprob_vbs(self):
        if len(self.hypotheses) > 1:
            logger = logging.getLogger('logger')
            for i in range(0, len(self.hypotheses), 2):
                # celbo1 - celbo0 (+ jump_bias)
                logger.info(f'celbo1: {self.hypotheses[i+1].neg_celbo}')
                logger.info(f'celbo0: {self.hypotheses[i].neg_celbo}')
                z = - self.hypotheses[i+1].neg_celbo + \
                    self.hypotheses[i].neg_celbo + self.config.jump_bias
                z = z.to(self.device)
                logger.info(f'z: {z}')
                z1 = -torch.log1p(torch.exp(-z))
                # log q(s_t=1) = log (sigmoid(z)) = log (1 / (1 + exp(-z)) =
                # -log(1+exp(-z))
                z0 = -torch.log1p(torch.exp(+z))
                # log q(s_t=0) = log (1 - q(s_t=1)) = log(1 - sigmoid(z)) =
                # log(sigmoid(-z)) = -log(1+exp(z))
                if z1 == -torch.inf:
                    z1 = z
                if z0 == -torch.inf:
                    z0 = -z
                self.hypotheses[i + 1].logprob = \
                    self.hypotheses[i+1].logprob + z1
                self.hypotheses[i].logprob = self.hypotheses[i].logprob + z0
                logger.info(f'z1 -- {z1}')
                logger.info(f'z0 -- {z0}')
        else:
            self.hypotheses[0].logprob = torch.Tensor([0.]).to(self.device)

    def __normalize_prob(self):
        if len(self.hypotheses) > 1:
            sum = torch.zeros_like(self.hypotheses[0].prob)
            for hypo in self.hypotheses:
                hypo.prob = torch.exp(hypo.logprob)
                sum += hypo.prob
            for hypo in self.hypotheses:
                hypo.prob /= sum
        else:
            self.hypotheses[0].prob = torch.Tensor([1.]).to(self.device)

    def prune_vbs(self):
        if len(self.hypotheses) > self.num_models:
            self.hypotheses = sorted(self.hypotheses,
                                     reverse=True,
                                     key=lambda hypo: hypo.logprob)
            self.hypotheses = self.hypotheses[:self.num_models]

    def update_vbs(self, X, y):
        self.__add_and_cal_prior_prob_vbs()

        if self.config.multiprocessing:
            with Pool(processes=self.config.cpus) as pool:
                self.hypotheses = pool.map(update_qz_job,
                                           zip(self.hypotheses,
                                               [X]*len(self.hypotheses),
                                               [y]*len(self.hypotheses)))
        else:
            for hypo in self.hypotheses:
                hypo.cal_qz(X, y)

        self.__update_logprob_vbs()
        if self.config.prune:
            self.prune_vbs()
        self.__normalize_prob()
        self.log_vbs()

    def log_vbs(self):
        logger = logging.getLogger("logger")
        for hypo in self.hypotheses:
            logger.info('history: {} --- logprob: {}'.format(hypo.history,
                                                             hypo.logprob.item()))

    def test_ensemble(self, X_test, y_test):
        with torch.no_grad():
            if self.config.variational_dropout:
                n_correct = torch.tensor(0, device=self.device)
                for batch_id in range(0, X_test.shape[0], self.batch_size):
                    X_test_batch = X_test[batch_id:batch_id+self.batch_size]
                    y_test_batch = y_test[batch_id:batch_id+self.batch_size]
                    y_pred = torch.zeros((X_test_batch.shape[0], 10),
                                        device=self.device)
                    for i in range(len(self.hypotheses)):
                        self.hypotheses[i].model.eval()
                        y_pred += self.hypotheses[i].model(X_test_batch, self.hypotheses[i].st)\
                            * self.hypotheses[i].prob
                        for _ in range(self.config.mcSamples_test-1):
                            y_pred += self.hypotheses[i].model(X_test_batch, self.hypotheses[i].st) \
                                * self.hypotheses[i].prob
                    y_pred = torch.argmax(y_pred, dim=-1)
                    n_correct += (y_pred == y_test_batch).sum()
            else:
                n_correct = torch.tensor(0, device=self.device)
                for batch_id in range(0, X_test.shape[0], self.batch_size):
                    X_test_batch = X_test[batch_id:batch_id+self.batch_size]
                    y_test_batch = y_test[batch_id:batch_id+self.batch_size]
                    y_pred = torch.zeros((X_test_batch.shape[0], 10),
                                        device=self.device)
                    for i in range(len(self.hypotheses)):
                        self.hypotheses[i].model.eval()
                        y_pred += self.hypotheses[i].model(X_test_batch)\
                            * self.hypotheses[i].prob
                        for _ in range(self.config.mcSamples_test-1):
                            y_pred += self.hypotheses[i].model(X_test_batch) \
                                * self.hypotheses[i].prob
                    y_pred = torch.argmax(y_pred, dim=-1)
                    n_correct += (y_pred == y_test_batch).sum()

            accuracy = n_correct/y_test.shape[0]
            logger = logging.getLogger("logger")
            logger.info(f'acc: {accuracy}')
        return accuracy

    def test_real_data(self, X_test, y_test, log_odds):
        abserr = self.hypotheses[0].test_abserr(X_test, y_test, log_odds)
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
