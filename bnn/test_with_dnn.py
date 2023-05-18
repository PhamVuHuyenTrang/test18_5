from data.DataProvider import DataProvider
from HypothesesStorageVBS_Hnet import HypothesesStorage
import numpy as np
import logging
from data.BatchGenerator import Batch
from config import config
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data.SVHN.SVHNBatch import FullSVHN


def get_train_data(current_task, device):
    x = torch.Tensor(current_task[0]).to(device)
    x.requires_grad_(False)
    y = torch.Tensor(current_task[1]).to(device)
    # y = F.one_hot(torch.Tensor(current_task[1]).to(torch.int64)) \
    #     .to(torch.float32).to(device)
    y.requires_grad_(False)
    return x, y


def get_test_data(next_task, device):
    x_test = torch.Tensor(next_task[0]).to(device)
    x_test.requires_grad_(False)
    y_test = torch.Tensor(next_task[1]).to(device)
    # y_test = F.one_hot(torch.Tensor(next_task[1]).to(torch.int64)) \
    #     .to(torch.float32).to(device)
    y_test.requires_grad_(False)
    return x_test, y_test


if __name__ == "__main__":
    config = config()
    torch.autograd.set_detect_anomaly(True)

    if config.device == 'whatever':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.device

    stream = FullSVHN(max_iter=2)
    logger = logging.getLogger('logger')
    logger.info('TEST WITH DNN!')
    accs = []
    current_task = None
    next_task = stream.get_chunk()

    while (not stream.is_dry()):
        current_task = next_task
        next_task = stream.get_chunk()

        x_train, y_train = get_train_data(current_task, device)
        x_test, y_test = get_test_data(next_task, device)

        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3))
        layers.append(nn.Conv2d(in_channels=32,
                      out_channels=32, kernel_size=3))
        layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Dropout2d(p=0.2))
        layers.append(nn.Conv2d(in_channels=32,
                      out_channels=64, kernel_size=3))
        layers.append(nn.Conv2d(in_channels=64,
                      out_channels=64, kernel_size=3))
        layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Dropout2d(p=0.2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1600, 10))

        net = nn.Sequential(*layers).to(device)

        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.Adam(params=list(net.parameters()),
                                     lr=config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                       step_size=200,
                                                       gamma=0.5)
        batch_size = config.batch_size
        for epoch_id in tqdm(range(config.epochs)):
            logger.info('Epoch: {}/{}'.format(epoch_id, config.epochs))
            epoch_loss = 0.
            for batch_id in range(0, x_train.shape[0], batch_size):
                X_batch = x_train[batch_id:batch_id+batch_size]
                y_batch = y_train[batch_id:batch_id+batch_size]
                optimizer.zero_grad()
                loss = loss_func(net(X_batch), y_batch)
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            logger.info('*****Loss: {}'.format(epoch_loss))

            if epoch_id % 50 == 49:
                torch.save(net.state_dict(),
                           f'DNNCheckpoint/epoch_{epoch_id+1}.pth')

                with torch.no_grad():
                    num_correct = 0
                    for batch_id in range(0, x_test.shape[0], batch_size):
                        X_batch = x_test[batch_id:batch_id+batch_size]
                        y_batch = y_test[batch_id:batch_id+batch_size]
                        y_pred = net(X_batch)
                        num_correct += (y_pred.argmax(dim=-1) ==
                                        y_batch.argmax(dim=-1)).float().sum()
                    acc = num_correct / x_test.shape[0]
                    logger.info('*****Test accuracy: {}'.format(acc))
