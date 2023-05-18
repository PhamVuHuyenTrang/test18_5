import copy
import torch
from torchvision import datasets


class PMNISTGenerator:
    def __init__(self, max_iter=100, task_size=10000, changerate=5, seed=1410, device='cpu'):
        mnist_trainset = datasets.MNIST(root='.data/MNIST/train', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='.data/MNIST/test', train=False, download=True, transform=None)

        self.X_train = mnist_trainset.data.to(device)/255.
        self.y_train = mnist_trainset.targets.to(device)
        # self.y_train = torch.nn.functional.one_hot(torch.Tensor(self.y_train)\
        #     .to(torch.int64)).to(torch.float32)
        self.X_test = mnist_testset.data.to(device)/255.
        self.y_test = mnist_testset.targets.to(device)
        # self.y_test = torch.nn.functional.one_hot(torch.Tensor(self.y_test)\
        #     .to(torch.int64)).to(torch.float32)
        self.cur_iter = -1
        self.device = device
        self.max_iter = max_iter
        self.task_size = task_size
        self.changerate = changerate
        self.seed = seed
        torch.manual_seed(seed)
        self.perm_list = [torch.randperm(28*28) for _ in range(round(max_iter/changerate + 1))]
        self.data_list = [torch.randperm(60000)[:task_size] for _ in range(max_iter)]

    def is_dry(self):
        return self.cur_iter >= self.max_iter - 1

    def get_chunk(self):
        assert False, 'Not to use!'
        self.cur_iter += 1
        X = copy.deepcopy(self.X[self.data_list[self.cur_iter]])
        y = copy.deepcopy(self.y[self.data_list[self.cur_iter]])
        X = X.view(self.task_size, 28*28)
        X = X[:, self.perm_list[self.cur_iter//self.changerate]]
        X = X.view(self.task_size, 28, 28)
        return X, y
    
    def next_task(self):
        self.cur_iter += 1
        X_train = copy.deepcopy(self.X_train[self.data_list[self.cur_iter]])
        y_train = copy.deepcopy(self.y_train[self.data_list[self.cur_iter]])
        X_train = X_train.view(self.task_size, 28*28)
        X_train = X_train[:, self.perm_list[self.cur_iter//self.changerate]]
        X_train = X_train.view(self.task_size, 28, 28)
        X_test = copy.deepcopy(self.X_test)
        y_test = copy.deepcopy(self.y_test)
        X_test = X_test.view(self.task_size, 28*28)
        X_test = X_test[:, self.perm_list[self.cur_iter//self.changerate]]
        X_test = X_test.view(self.task_size, 28, 28)
        return X_train, y_train, X_test, y_test


class FullMNIST:
    def __init__(self, max_iter=100, task_size=10000, changerate=5, seed=1410, device='cpu'):
        mnist_trainset = datasets.MNIST(root='.data/MNIST/train', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='.data/MNIST/test', train=False, download=True, transform=None)

        self.X_train = mnist_trainset.data.to(device)/255.
        self.y_train = mnist_trainset.targets.to(device)
        # self.y_train = torch.nn.functional.one_hot(torch.Tensor(self.y_train) \
        #     .to(torch.int64)).to(torch.float32)
        self.X_test = mnist_testset.data.to(device)/255.
        self.y_test = mnist_testset.targets.to(device)
        # self.y_test = torch.nn.functional.one_hot(torch.Tensor(self.y_test) \
        #     .to(torch.int64)).to(torch.float32)
        self.cur_iter = -1
        self.device = device
        self.max_iter = max_iter
        self.task_size = task_size
        self.changerate = changerate
        self.seed = seed
        torch.manual_seed(seed)

    def is_dry(self):
        return self.cur_iter >= self.max_iter - 1

    def get_chunk(self):
        assert False, 'Not to use!'
        self.cur_iter += 1
        X = copy.deepcopy(self.X[self.data_list[self.cur_iter]])
        y = copy.deepcopy(self.y[self.data_list[self.cur_iter]])
        X = X.view(self.task_size, 28*28)
        X = X[:, self.perm_list[self.cur_iter//self.changerate]]
        X = X.view(self.task_size, 28, 28)
        return X, y

    def next_task(self):
        self.cur_iter += 1
        return self.X_train, self.y_train, self.X_test, self.y_test
