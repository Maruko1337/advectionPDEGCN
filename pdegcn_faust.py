
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# from torch_geometric.utils import grid
from torch_geometric.datasets import ModelNet, FAUST
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

base_path = './'
import graphOps as GO
import utils
import graphNet as GN

# Setup the network and its parameters
nNin = 6  # 6  # 6
nEin = 256  # 3
nopen = 256  # 64
nhid = 256  # 64
nNclose = 256  # 64
nlayer = 8  # 8#16

batchSize = 32
h = 0.01
lr = 0.01
lrGCN = 0.001
wdGCN = 0
wd = 0

faust_path = ''
transforms = T.FaceToEdge(remove_faces=False)


pre_transform = T.Compose([T.FaceToEdge(remove_faces=False), T.Constant(value=1)])
train_dataset = FAUST(faust_path, True, T.Cartesian(), pre_transform)
test_dataset = FAUST(faust_path, False, T.Cartesian(), pre_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1)
d = train_dataset[0]

model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=True,
                                  diffOrder=1, num_output=d.num_nodes, dropOut=0.0, faust=True,
                                  gated=False,
                                  realVarlet=False, mixDyamics=True)

model.to(device)

target = torch.arange(d.num_nodes, dtype=torch.long, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train(epoch):
    model.train()

    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    total_loss = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        I = data.edge_index[0, :]
        J = data.edge_index[1, :]
        N = data.pos.shape[0]
        W = torch.ones(N).to(device)
        G = GO.graph(I, J, N, W=W, pos=data.pos, faces=data.face.t())
        G = G.to(device)
        xn = data.x.t().unsqueeze(0)
        xn = data.pos.t().unsqueeze(0)
        xe = data.edge_attr.t().unsqueeze(0)

        [xnOut, beta] = model(xn, G, xe=xe)
        # print("beta:", beta)
        loss = F.nll_loss(xnOut, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 100 == 9:
            print("train loss:", total_loss / 10)
            total_loss = 0


acc_hist = []


def test():
    model.eval()
    correct = 0

    for idx, data in enumerate(test_loader):
        data = data.to(device)
        optimizer.zero_grad()

        I = data.edge_index[0, :]
        J = data.edge_index[1, :]
        N = data.pos.shape[0]
        W = torch.ones(N).to(device)
        G = GO.graph(I, J, N, W=W, pos=data.pos, faces=data.face.t())
        G = G.to(device)
        xn = data.x.t().unsqueeze(0)
        xn = data.pos.t().unsqueeze(0)
        xe = data.edge_attr.t().unsqueeze(0)
        [xnOut, beta] = model(xn, G, xe=xe)
        # if idx == 0:
        #     betas.append(beta)
        pred = xnOut.max(1)[1]
        correct += pred.eq(target).sum().item()
    return correct / (len(test_dataset) * d.num_nodes) * 100


for epoch in range(1, 1001):
    train(epoch)
    test_acc = test()
    acc_hist.append(test_acc)
    if (epoch % 1 == 0):
        print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
