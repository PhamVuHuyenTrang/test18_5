from data.dataloader import *
from HypothesesStorageVBS_hnet import HypothesesStorage
import numpy as np
import logging
from config import config
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import time
from data.DataProvider import BatchDivider


if __name__ == "__main__":
    config = config()

    if config.device == 'whatever':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.device

    if config.dataset == "elec2":
        X, y, X_test_set, y_test_set = get_elec2_dataset(valid=False)
        log_odds = True
        num_feature = 15
    elif config.dataset == "sensordrift":
        X, y = get_sensordrift_dataset(valid=False)
        log_odds = False
        num_feature = 129
    elif config.dataset == "malware":
        X, y, X_test_set, y_test_set = get_malware_dataset(valid=False)
        log_odds = True
        num_feature = 483

    X = torch.from_numpy(X).to(device).type(torch.float32)
    y = torch.from_numpy(y).to(device).type(torch.float32)

    HS = HypothesesStorage(config=config, device=device)

    datagen = BatchDivider(X,y,mini_batch_size=50)

    logger = logging.getLogger('logger')
    # print(batch.chunk_id)
    # print(batch.n_chunks)
    task_id = 0
    mcaes = []
    while (task_id < 101):
        X_train, y_train, X_test, y_test = datagen.next_task()
        begin_time = time.time()
        X_train = X_train.reshape((X_train.shape[0], num_feature))
        y_train = y_train.reshape((X_train.shape[0], 1))
        X_test = X_test.reshape((X_test.shape[0], num_feature))
        y_test = y_test.reshape((X_test.shape[0], 1))
        if task_id == 0:
            HS.update_vbs(X_train, y_train, epochs = 1)
        else:
            HS.update_vbs(X_train, y_train, epochs = 1)
        HS.test_real_data(X_test, y_test, log_odds)
        logger.info(f'\n********** TASK {task_id} **********')
        mcae = HS.test_abserr[task_id]
        print("HS.test_abserr", HS.test_abserr)
        #print(mcae)
        #print("Number of executed tasks: ")
        #print(len(HS.test_abserr))
        logger.info(f'MCAE: {mcae}')
        logger.info(f'finish time: {datetime.datetime.now()}')
        print("done")
        task_id += 1
