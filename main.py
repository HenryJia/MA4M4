import numpy as np
import networkx as nx
import torch
from torch import optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from vbsbm import VBSBM

plt.ion()

G_karate = nx.read_edgelist('ucidata-zachary/out.ucidata-zachary', comments='%', nodetype=int)

plt.figure()
pos = nx.spring_layout(G_karate, seed=1234)
nx.draw(G_karate, with_labels=True, pos=pos)

A_karate = torch.from_numpy(nx.to_numpy_array(G_karate))

K = 2
# Set our priors
mu = torch.zeros(A_karate.shape[0], K)
# Steer things with the prior :)
#mu[33, 0] = 10
#mu[30, 0] = -10
tau = torch.eye(K) * torch.mean(A_karate)
sigma = torch.tensor(0.1)

model = VBSBM(mu, tau, sigma, use_vi=True)

#model.cuda()
#A_karate = A_karate.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
pb = tqdm()
for i in range(10000):
    optimizer.zero_grad()
    elbo = -model(A_karate)
    elbo.backward()
    optimizer.step()
    pb.set_postfix({'elbo': elbo.item(), 'kl': model.kl().item()})
    pb.update(1)
pb.close()

colors = (np.linspace(0, 1, K) * 20 + 10)
c, A = model.sample()
c = torch.argmax(model.c, dim=1).numpy()
A = A.numpy()

A = A * (1 - np.eye(A.shape[0])) # Remove self loops, we know this graph has no self loops
for i in range(A.shape[0]): # And make symmetric
    for j in range(A.shape[1]):
        A[i, j] = float(A[i, j] > 0 or A[j, i] > 0)

plt.figure()
pos = nx.spring_layout(G_karate, seed=1234)
nx.draw(G_karate, with_labels=True, node_color=np.array([colors[block] for block in c]), pos=pos)
