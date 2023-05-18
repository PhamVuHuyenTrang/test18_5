import torch
import time

# experiment: sampling on cuda vs sampling on cpu and transfer to gpu


begin_time = time.time()
x = torch.rand((100, 100, 100))
print(f'sample time: {time.time()-begin_time}')
begin_time = time.time()
x = x.to('cuda')
print(f'change device time: {time.time()-begin_time}')
begin_time = time.time()
y = x**2
print(f'calculation time: {time.time() - begin_time}')
time.sleep(10)

del x, y
print(f'sample on gpu***')

begin_time = time.time()
shape = torch.Size((100, 100, 100))
x = torch.cuda.FloatTensor(shape)
torch.rand(shape, out=x)
print(f'sample time: {time.time()-begin_time}')
begin_time = time.time()
y = x**2
print(f'calculation time: {time.time()-begin_time}')
time.sleep(10)
