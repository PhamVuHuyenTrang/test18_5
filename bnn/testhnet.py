import numpy as np
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# from hypnettorch.data import FashionMNISTData, MNISTData

data_dir = '.'

# mnist = MNISTData(data_dir, use_one_hot=True, validation_size=0)
# fmnist = FashionMNISTData(data_dir, use_one_hot=True, validation_size=0)

# Get a batch of training samples from each data handler.
# mnist_inps, mnist_trgts = mnist.next_train_batch(4)
# fmnist_inps, fmnist_trgts = fmnist.next_train_batch(4)

# mnist.plot_samples('MNIST Examples', mnist_inps, outputs=mnist_trgts)
# fmnist.plot_samples('FashionMNIST Examples', fmnist_inps, outputs=fmnist_trgts)

from mnets import LeNet

mnet = LeNet(in_shape=[28, 28], num_classes=2,
             arch='mnist_small', no_weights=True).to(device)

from hnets import HMLP

hnet = HMLP(mnet.param_shapes, uncond_in_size=0, cond_in_size=8,
            layers=[100, 100], num_cond_embs=2).to(device)

for param in hnet.parameters():
    print(param)

exit()
print()
print('The randomly initialized input embeddings are:\n', 
      hnet.conditional_params)

# To produce main network weights for condition `0`, we can either pass
# the corresponding condition ID, or the corresponding (internally maintained)
# embedding to the `forward` of the hypernetwork.
W0 = hnet.forward(cond_id=0)
W0_tmp = hnet.forward(cond_input=hnet.conditional_params[0].view(1, -1))
assert np.all([torch.equal(W0[i], W0_tmp[i]) for i in range(len(W0))])

# Hypernetworks also allow batch processing.
W_batch = hnet.forward(cond_id=[0, 0])
assert np.all([torch.equal(W_batch[0][i], W_batch[1][i]) \
               for i in range(len(W0))])


def calc_accuracy(data, mnet, mnet_weights):
    """Compute the test accuracy for a given dataset"""
    with torch.no_grad():
        # Process complete test set as one batch.
        test_in = data.input_to_torch_tensor( \
            data.get_test_inputs(), device, mode='inference')
        test_out = data.input_to_torch_tensor( \
            data.get_test_outputs(), device, mode='inference')
        test_lbls = test_out.max(dim=1)[1]

        logits = mnet(test_in, weights=mnet_weights)
        pred_lbls = logits.max(dim=1)[1]

        acc = torch.sum(test_lbls == pred_lbls) / test_lbls.numel() * 100.

    return acc

# Configure training.
lr=1e-4
batchsize=32
nepochs=0

# Adam usually works well in combination with hypernetwork training.
optimizer = torch.optim.Adam(hnet.internal_params, lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(nepochs): # For each epoch.

    # Iterate over the whole MNIST/FashionMNIST training set.
    # Note, that both datasets have the same number of training samples.
    i = 0
    for curr_batchsize, mx, my in mnist.train_iterator(batchsize):
        i += 1

        # Current mini-batch of MNIST samples.
        mnist_X = mnist.input_to_torch_tensor(mx, device, mode='train')
        mnist_Y = mnist.output_to_torch_tensor(my, device, mode='train')
        
        # Current mini-batch of FashionMNIST samples.
        fx, fy = fmnist.next_train_batch(curr_batchsize)
        fmnist_X = fmnist.input_to_torch_tensor(fx, device, mode='train')
        fmnist_Y = fmnist.output_to_torch_tensor(fy, device, mode='train')

        optimizer.zero_grad()

        # Compute MNIST loss.
        W_mnist = hnet(cond_id=0)
        mnist_P = mnet.forward(mnist_X, weights=W_mnist)
        loss_mnist = criterion(mnist_P, mnist_Y.max(dim=1)[1])
        
        # Compute FashionMNIST loss.
        W_fmnist = hnet(cond_id=1)
        fmnist_P = mnet.forward(fmnist_X, weights=W_fmnist)
        loss_fmnist = criterion(fmnist_P, fmnist_Y.max(dim=1)[1])
        
        # The total loss is simply each task's loss combined.
        loss = loss_mnist + loss_fmnist
        loss.backward()
        optimizer.step()

        if i % 500 == 0:            
            print('[%d, %5d] loss: %.3f, MNIST acc: %.2f%%, FashionMNIST acc: %.2f%%' %
                  (epoch + 1, i + 1, loss.item(), 
                   calc_accuracy(mnist, mnet, W_mnist),
                   calc_accuracy(fmnist, mnet, W_fmnist)))


print('Training finished with test-accs: MNIST acc: %.2f%%, FashionMNIST %.2f%%' % \
      (calc_accuracy(mnist, mnet, W_mnist),
       calc_accuracy(fmnist, mnet, W_fmnist)))
