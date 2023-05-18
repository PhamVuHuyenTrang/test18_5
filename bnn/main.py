from data.DataProvider import DataProvider
from HypothesesStorageVBS_hnet import HypothesesStorage
import numpy as np
import logging
from data.BatchGenerator import Batch
import os
from config import config
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data.toydata.ToyBatch import ToyBatch


def get_train_data(current_task, device):
    x = torch.Tensor(current_task[0]).to(device)
    x.requires_grad_(False)
    y = torch.Tensor(current_task[1]).to(torch.int64).to(device)
    # y = F.one_hot(torch.Tensor(current_task[1]).to(torch.int64)) \
    #     .to(torch.float32).to(device)
    y.requires_grad_(False)
    return x, y


def get_test_data(next_task, device):
    x_test = torch.Tensor(next_task[0]).to(device)
    x_test.requires_grad_(False)
    y_test = torch.Tensor(next_task[1]).to(torch.int64).to(device)
    # y_test = F.one_hot(torch.Tensor(next_task[1]).to(torch.int64)) \
    #     .to(torch.float32).to(device)
    y_test.requires_grad_(False)
    return x_test, y_test


if __name__ == "__main__":
    torch.manual_seed(0)
    config = config()
    torch.set_printoptions(precision=10)
    torch.autograd.set_detect_anomaly(True)

    if config.device == 'whatever':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.device

    if config.dataset == 'svhn':
        if config.kaggle:
            if config.save_model:
                from data.SVHN.SVHNBatch import FullSVHNForKaggle
                stream = FullSVHNForKaggle(max_iter=config.max_iter,
                                           changerate=config.changerate)
            else:
                from data.SVHN.SVHNBatch import LongTransformedSvhnGeneratorForKaggle
                stream = LongTransformedSvhnGeneratorForKaggle(
                    max_iter=config.max_iter)
        elif config.ggcolab:
            if config.save_model:
                from data.SVHN.SVHNBatch import FullSVHNForGGColab
                stream = FullSVHNForGGColab(max_iter=config.max_iter,
                                            changerate=config.changerate)
            else:
                from data.SVHN.SVHNBatch import LongTransformedSvhnGeneratorForGGColab
                stream = LongTransformedSvhnGeneratorForGGColab(
                    max_iter=config.max_iter)
        else:
            if config.save_model:
                from data.SVHN.SVHNBatch import FullSVHN
                stream = FullSVHN(max_iter=config.max_iter,
                                  changerate=config.changerate)
            else:
                from data.SVHN.SVHNBatch import LongTransformedSvhnGenerator
                stream = LongTransformedSvhnGenerator(max_iter=config.max_iter)
    elif config.dataset == 'permutedMNIST':
        if config.save_model:
            from data.PermutedMNIST.PermutedMNISTBatch import FullMNIST
            stream = FullMNIST(max_iter=config.max_iter, device=device)
        else:
            from data.PermutedMNIST.PermutedMNISTBatch import PMNISTGenerator
            stream = PMNISTGenerator(
                max_iter=config.max_iter, device=device, changerate=config.changerate)
            
    elif config.dataset == 'cifar':
        if config.save_model:
            from data.CIFAR.CIFARBatch import FullCIFAR
            stream = FullCIFAR(max_iter=config.max_iter)
        else:
            from data.CIFAR.CIFARBatch import LongTransformedCifar10Generator
            stream = LongTransformedCifar10Generator(
                max_iter=config.max_iter, changerate=config.changerate)
    elif config.dataset == 'toy':
        stream = ToyBatch(changerate=5, task_size=100, max_iter=100)

    try:
        HS = HypothesesStorage(config, device)

    except Exception:
        HS = HypothesesStorage(config, device, None, None)

    logger = logging.getLogger('logger')
    accs = []
    # current_task = None
    # next_task = stream.get_chunk()

    while (not stream.is_dry()):
        X_train, y_train, X_test, y_test = stream.next_task()
        if type(X_train) == np.ndarray:
            X_train = torch.from_numpy(X_train).to(device)
            y_train = torch.Tensor(y_train).to(torch.int64).to(device)
            X_test = torch.from_numpy(X_test).to(device)
            y_test = torch.Tensor(y_test).to(torch.int64).to(device)
        # current_task = next_task
        # next_task = stream.get_chunk()

        # X_train, y_train = get_train_data(current_task, device)
        # X_test, y_test = get_test_data(next_task, device)

        logger.info(f'********** TASK {stream.cur_iter} **********')
        HS.update_vbs(X_train.to(device), y_train.to(device))
        acc = HS.test_ensemble(X_test.to(device), y_test.to(device))
        print(f'task: {stream.cur_iter}, acc: {acc}')
        accs.append(acc)
        if config.save_model:
            if not config.hnet:
                path = f'checkpoint/{config.dataset}/{"dropout" if config.variational_dropout else "vbs"}/'
                if not os.path.isdir(path):
                    os.makedirs(path)
                torch.save(HS.hypotheses[0].model.state_dict(), path + f'epoch{stream.cur_iter}.pt')
            else:
                path = f'checkpoint/{config.dataset}/hnet/'
                if not os.path.isdir(path):
                    os.makedirs(path)
                torch.save(HS.hnet.state_dict(),
                           path + f'epoch{stream.cur_iter}.pt')
                torch.save(HS.hypotheses[0].task_embedding,
                        path + f'hnet_task_embedding{stream.cur_iter}.pt')
