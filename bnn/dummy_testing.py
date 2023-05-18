import torch
import probabilistic
import hnets
import mnets

mnet = mnets.MLP(3, 5, [4, 6], use_bias=True)
mnet = probabilistic.gauss_mnet_interface.GaussianBNNWrapper(mnet)
hnet = hnets.HMLP(mnet.param_shapes, cond_in_size=8,
                       layers=[1, 1])
hnet.apply_hyperfan_init()
optimizer = torch.optim.Adam(hnet.parameters(), lr=0.01)
for _ in range(10):
    optimizer.zero_grad()
    emb1 = torch.randn((1, 8))*10
    emb2 = torch.randn((1, 8))*10
    X = torch.randn((1, 3))
    weight_mnet1 = hnet.forward(cond_input=emb1)
    y = mnet.forward(X, weight_mnet1)
    weight_mnet2 = hnet.forward(cond_input=emb2)
    y += mnet.forward(X, weight_mnet2)

    loss = (y**2).sum()
    loss.backward()
    optimizer.step()
    