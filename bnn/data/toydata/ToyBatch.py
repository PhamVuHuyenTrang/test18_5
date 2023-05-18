import torch
import numpy as np

class ToyBatch:
    def __init__(self, changerate, task_size, max_iter):
        self.changerate = changerate
        self.task_size = task_size
        self.max_iter = max_iter
        self.cur_iter = 0
        self.meanA = torch.tensor([0.0, 0.0])
        self.meanB = torch.tensor([1.0, 1.0])
        self.sigmaA = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.sigmaB = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def is_dry(self):
        return (self.cur_iter >= self.max_iter)

    def next_task(self):
        self.task = torch.randint(0, 2, (self.task_size, 1))
        self.iter = 0
        if self.cur_iter % self.changerate == 0:
            self.meanA = torch.randn((2))
            self.meanB = torch.randn((2))

        self.X_trainA = np.random.multivariate_normal(self.meanA, self.sigmaA, (self.task_size//2))
        self.X_trainB = np.random.multivariate_normal(self.meanB, self.sigmaB, (self.task_size//2))

        self.X_trainA = torch.from_numpy(self.X_trainA)
        self.X_trainB = torch.from_numpy(self.X_trainB)
        self.y_trainA = torch.zeros((self.task_size, 1))
        self.y_trainB = torch.ones((self.task_size, 1))

        shuffle = torch.randperm(self.task_size)

        self.X_train = torch.cat((self.X_trainA, self.X_trainB), dim=0)[shuffle]
        self.y_train = torch.cat((self.y_trainA, self.y_trainB), dim=0)[shuffle]

        
        self.X_testA = np.random.multivariate_normal(self.meanA, self.sigmaA, (self.task_size//2))
        self.X_testB = np.random.multivariate_normal(self.meanB, self.sigmaB, (self.task_size//2))
        self.X_testA = torch.from_numpy(self.X_testA).to(torch.float32)
        self.X_testB = torch.from_numpy(self.X_testB).to(torch.float32)
        self.y_testA = torch.zeros((self.task_size, 1))
        self.y_testB = torch.ones((self.task_size, 1))

        shuffle = torch.randperm(self.task_size)

        self.X_test = torch.cat((self.X_testA, self.X_testB), dim=0)[shuffle]
        self.y_test = torch.cat((self.y_testA, self.y_testB), dim=0)[shuffle]

        return self.X_train, self.y_train, self.X_test, self.y_test