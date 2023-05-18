from argparse import Namespace
import torch
import torch.nn as nn
import logging
import multiprocessing
from tqdm import tqdm
from BNN.Model import Model
from copy import deepcopy
import time
import numpy as np
from multiprocessing.dummy import Pool


class Hypothesis:
    def __init__(self, config: Namespace, device, num_feature):
        self.config = config
        self.device = device
        self.run_length = 0
        self.history = ""
        self.prob = torch.Tensor([self.config.Lambda]).to(self.device)
        self.logprob = torch.Tensor([0.])  # used for vbs
        self.prob.requires_grad_(False)
        self.model = Model(self.config, self.run_length, in_size=num_feature, device=self.device)
        self.neg_celbo = None
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = lambda x, y: ((x-y)**2/(2*config.noise**2)).mean()

    def cal_qz(self, X, y):
        tmp_model = Model.clone(self.model)
        tmp_model.pertube()
        tmp_model.set_trainable(True)

        optimizer = torch.optim.Adam(params=tmp_model.parameters(),
                                     lr=self.config.learning_rate)
        epochs = self.config.epochs
        tmp_model.train()
        # begin_time = time.time()
        for _ in tqdm(range(epochs)):
            neg_celbo = self.__cal_neg_celbo(X, y, tmp_model)
            optimizer.zero_grad()
            neg_celbo.backward()
            optimizer.step()
        # print(f'1 hypothesis training time: {time.time()-begin_time}')

        tmp_model.set_trainable(False)
        self.neg_celbo = self.__cal_neg_celbo(X, y, tmp_model, finalround=True)
        self.model = tmp_model
        self.model.eval()

    def __cal_neg_celbo(self, X, y, tmp_model: Model, finalround=False):
        if not finalround:
            z = [tmp_model(X) for _ in range(self.config.mcSamples)]
        else:
            z = [tmp_model(X) for _ in range(self.config.mcSamples_finalround)]
        neg_log_likelihood_samples = [self.loss(zz, y) for zz in z]
        neg_log_likelihood = torch.mean(torch.stack(neg_log_likelihood_samples))
        KL = Model.kl_divergence(tmp_model, self.model)
        neg_celbo = neg_log_likelihood + KL
        return neg_celbo

    def test_abserr(self, X_test, y_test, log_odds):  # for real_data
        list_y_pred = [self.model(X_test) for _ in range(self.config.mcSamples_test)]
        if log_odds:
            list_y_pred = [torch.sigmoid(y_pred) for y_pred in list_y_pred]  # log-odds
        y_pred = sum(list_y_pred) / len(list_y_pred)
        return torch.abs(y_pred-y_test).mean()

    def increase_runlength(self):
        self.run_length += 1
        self.model.increase_runlength()

    @classmethod
    def clone(cls, hypo):
        new_hypo = Hypothesis(hypo.config, hypo.device, hypo.model.in_size)
        new_hypo.history = hypo.history
        new_hypo.loss = hypo.loss
        new_hypo.model = Model.clone(hypo.model)
        new_hypo.neg_celbo = None
        new_hypo.prob = hypo.prob
        return new_hypo


class HypothesesStorage():
    def __init__(self, config: Namespace, device, num_feature):
        self.device = device
        self.config = config
        self.hypotheses = []
        self.num_models = self.config.num_models  # for pruning
        self.Lambda = self.config.Lambda
        self.test_abserr = []  # for real data
        self.num_feature = num_feature

    def __add_and_cal_prior_prob(self):
        for hypo in self.hypotheses:
            hypo.increase_runlength()
            hypo.prob = (1-self.Lambda)*hypo.prob
        self.hypotheses = [Hypothesis(self.config, self.device, self.num_feature)]\
            + self.hypotheses
        if len(self.hypotheses) == 1:
            self.hypotheses[0].prob /= self.hypotheses[0].prob

    def __update_prob(self):
        for hypo in self.hypotheses:
            hypo.prob = hypo.prob * torch.exp(-hypo.neg_celbo)

    def update(self, X, y):
        self.__add_and_cal_prior_prob()

        if self.config.multiprocessing:
            with Pool(processes=self.config.cpus) as pool:
                self.hypotheses = pool.map(update_qz_job,
                                           zip(self.hypotheses,
                                               [X]*len(self.hypotheses),
                                               [y]*len(self.hypotheses)))
        else:
            for hypo in self.hypotheses:
                hypo.cal_qz(X, y)

        self.__update_prob()

        if self.config.prune:
            self.prune()

        self.__renormalize_prob()

        self.log()

    def log(self):
        logger = logging.getLogger("logger")
        for hypo in self.hypotheses:
            logger.info('rl: {} --- prob: {}'.format(hypo.run_length,
                                                     hypo.prob.item()))

    def __renormalize_prob(self):
        if len(self.hypotheses) == 1:
            self.hypotheses[0].prob = torch.ones_like(self.hypotheses[0].prob)
            return
        sum = torch.zeros_like(self.hypotheses[0].prob).to(self.device)
        sum.requires_grad_(False)
        for hypo in self.hypotheses:
            sum += hypo.prob
        for hypo in self.hypotheses:
            hypo.prob = hypo.prob / sum

    def prune(self):
        if len(self.hypotheses) > self.num_models:
            self.hypotheses = sorted(self.hypotheses,
                                     reverse=True,
                                     key=lambda hypo: hypo.prob)
            self.hypotheses = self.hypotheses[:self.num_models]

    def __add_and_cal_prior_prob_vbs(self):
        new_hypotheses = []
        if len(self.hypotheses) > 0:
            for hypo in self.hypotheses:
                hypo_s0 = Hypothesis.clone(hypo)
                hypo_s0.run_length = 0
                hypo_s0.history += "0"
                hypo_s1 = Hypothesis.clone(hypo)
                hypo_s1.run_length = 1
                hypo_s1.history += "1"
                hypo_s1.model.broaden(1.5)
                new_hypotheses += [hypo_s0, hypo_s1]
            self.hypotheses = new_hypotheses
        else:
            hypo_s0 = Hypothesis(num_feature=self.num_feature, config=self.config, device=self.device)
            hypo_s0.run_length = 0
            hypo_s0.history += "0"
            hypo_s1 = Hypothesis(num_feature=self.num_feature, config=self.config, device=self.device)
            hypo_s1.run_length = 1
            hypo_s1.history += "1"
            self.hypotheses.append(hypo_s0)
            self.hypotheses.append(hypo_s1)

    def __update_logprob_vbs(self):
        for i in range(0, len(self.hypotheses), 2):
            # celbo1 - celbo0 (+ jump_bias)
            z = self.hypotheses[i+1].neg_celbo - self.hypotheses[i].neg_celbo
            print(z)
            # z /= 20
            # z -= 0.5
            z1 = -torch.log1p(torch.exp(-z))
            z0 = -torch.log1p(torch.exp(z))
            if z1 == -torch.inf:
                z1 = z
            if z0 == -torch.inf:
                z0 = -z
            self.hypotheses[i+1].logprob += z1
            self.hypotheses[i].logprob += z0

    def __normalize_prob(self):
        sum = torch.zeros_like(self.hypotheses[0].prob)
        for hypo in self.hypotheses:
            hypo.prob = torch.exp(hypo.logprob)
            sum += hypo.prob
        for hypo in self.hypotheses:
            hypo.prob /= sum

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
        for i in range(len(self.hypotheses)):
            y_pred = self.hypotheses[i].model(X_test)*self.hypotheses[i].prob
            for _ in range(self.config.mcSamples_test-1):
                y_pred += self.hypotheses[i].model(X_test) \
                    * self.hypotheses[i].prob
        y_pred /= self.config.mcSamples_test
        y_pred = y_pred.argmax(dim=-1)
        y_test_idx = y_test.argmax(dim=-1)
        accuracy = (y_pred == y_test_idx).sum()/y_test_idx.shape[0]
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
