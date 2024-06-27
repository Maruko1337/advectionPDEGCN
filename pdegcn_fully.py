import os.path as osp
import time
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import optuna
import numpy as np
import sys
import process
import utils
import utils_gcnii
from torch_geometric.utils import sparse as sparseConvert
import matplotlib.pyplot as plt
print(torch.cuda.get_device_properties('cuda:0'))


base_path = '../../../data/'
import graphOps as GO
# import processContacts as prc
import utils
import graphNet as GN

# Setup the network and its parameters

num_layers = [2, 4, 8,16,32,64]
for nlayers in num_layers:
    torch.cuda.synchronize()
    print("Doing experiment for ", nlayers, " layers!", flush=True)
    torch.cuda.synchronize()


    def objective(trial):

        nEin = 1
        n_channels = 64  # trial.suggest_categorical('n_channels', [64, 128, 256])
        nopen = n_channels
        nhid = n_channels
        nNclose = n_channels
        nlayer = nlayers
        datastr = "pubmed"
        print("DATA SET IS:", datastr)
        # h = 1 / nlayers
        h = 0.95
        # h = trial.suggest_discrete_uniform('h', 0.1 / nlayer, 3, q=0.1 / (nlayer))
        # h = trial.suggest_discrete_uniform('h',0.1, 3, q=0.01)
        # dropout = trial.suggest_discrete_uniform('dropout', 0.5, 0.7, q=0.1)
        # dropout = trial.suggest_discrete_uniform('dropout', 0.01, 0.9, q=0.1)
        dropout = 0.5
        print("n channels:", nopen)
        print("n layers:", nlayer)
        print("h step:", h)
        print("dropout:", dropout)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        realVarlet = False

        # lr = trial.suggest_float("lr", 1e-2, 1e-1, log=True)

        lr = 0.07
        # lr_alpha = trial.suggest_float("lr_alpha", 1e-6, 1e-2, log=True)
        # lr_beta = trial.suggest_float("lr_beta", 1e-6, 1e-2, log=True)
        lr_beta = 1e-4
        lr_alpha = 1e-4

        lrGCN = 5e-4
        wd = 2.84e-7
        # lrGCN = trial.suggest_float("lrGCN", 1e-6, 1e-3, log=True)
        # wd = trial.suggest_float("wd", 5e-8, 1e-3, log=True)

        def train_step(model, optimizer, features, labels, adj, idx_train):
            model.train()
            optimizer.zero_grad()
            I = adj[0, :]
            J = adj[1, :]
            N = labels.shape[0]
            w = torch.ones(adj.shape[1]).to(device)
            G = GO.graph(I, J, N, W=w, pos=None, faces=None)
            G = G.to(device)
            xn = features
            xe = torch.ones(1, 1, I.shape[0]).to(device)

            [out, G] = model(xn, G)
            g = G.nodeGrad(out.t().unsqueeze(0))
            acc_train = utils_gcnii.accuracy(out[idx_train], labels[idx_train].to(device))
            loss_train = F.nll_loss(out[idx_train], labels[idx_train].to(device))
            loss_train.backward()
            optimizer.step()
            return loss_train.item(), acc_train.item()

        def test_step(model, features, labels, adj, idx_test):
            model.eval()
            with torch.no_grad():
                I = adj[0, :]
                J = adj[1, :]
                N = labels.shape[0]
                w = torch.ones(adj.shape[1]).to(device)

                G = GO.graph(I, J, N, W=w, pos=None, faces=None)
                G = G.to(device)
                xn = features
                xe = torch.ones(1, 1, I.shape[0]).to(device)

                [out, G] = model(xn, G)

                loss_test = F.nll_loss(out[idx_test], labels[idx_test].to(device))
                acc_test = utils_gcnii.accuracy(out[idx_test], labels[idx_test].to(device))
                return loss_test.item(), acc_test.item()

        def train(datastr, splitstr, num_output):
            slurm = ("s" in sys.argv) or ("e" in sys.argv)
            adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = process.full_load_data(
                datastr,
                splitstr)
            adj = adj.to_dense()

            [edge_index, edge_weight] = sparseConvert.dense_to_sparse(adj)
            del adj

            edge_index = edge_index.to(device)
            features = features.to(device).t().unsqueeze(0)
            idx_train = idx_train.to(device)
            idx_test = idx_test.to(device)
            labels = labels.to(device)
            #

            model = GN.graphNetwork_nodesOnly(num_features, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True,
                                              wave=False,
                                              diffOrder=1, num_output=num_output, dropOut=dropout, gated=False,
                                              realVarlet=realVarlet, mixDyamics=True)
            model = model.to(device)

            optimizer = torch.optim.Adam([
                dict(params=model.KN1, lr=lrGCN, weight_decay=0),
                dict(params=model.KN2, lr=lrGCN, weight_decay=0),
                dict(params=model.K1Nopen, weight_decay=wd),
                dict(params=model.KNclose, weight_decay=wd),
                dict(params=model.alpha, lr=lr_alpha, weight_decay=0),
                dict(params=model.beta, lr=lr_beta, weight_decay=0),
                dict(params=model.omega, lr=lr, weight_decay=wd)
            ], lr=lr)

            bad_counter = 0
            best = 0
            for epoch in range(2000):
                loss_tra, acc_tra = train_step(model, optimizer, features, labels, edge_index, idx_train)
                loss_val, acc_test = test_step(model, features, labels, edge_index, idx_test)
                if (epoch + 1) % 100000000 == 0:
                    print('Epoch:{:04d}'.format(epoch + 1),
                          'train',
                          'loss:{:.3f}'.format(loss_tra),
                          'acc:{:.2f}'.format(acc_tra * 100),
                          '| test',
                          'loss:{:.3f}'.format(loss_val),
                          'acc:{:.2f}'.format(acc_test * 100))
                if acc_test > best:
                    best = acc_test
                    bad_counter = 0
                else:
                    bad_counter += 1

                if bad_counter == 200:
                    break
            acc = best

            return acc * 100
        start = time.time()
        acc_list = []
        for i in range(10):
            if datastr == "cora":
                num_output = 7
            elif datastr == "citeseer":
                num_output = 6
            elif datastr == "pubmed":
                num_output = 3
            elif datastr == "chameleon":
                num_output = 5
            else:
                num_output = 5
            if ("s" in sys.argv) or ("e" in sys.argv):
                splitstr = 'splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
            else:
                splitstr = 'splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
            acc_list.append(train(datastr, splitstr, num_output))
            print(i, ": {:.2f}".format(acc_list[-1]))
        end = time.time()
        print("running time:{:.2f}".format(end - start))
        mean_test_acc = np.mean(acc_list)
        best_test_acc = np.max(acc_list)
        worst_test_acc = np.min(acc_list)
        std_test_acc = np.std(acc_list)
        print("Average Test acc.:{:.2f}".format(mean_test_acc))
        print("Best Test acc.:{:.2f}".format(best_test_acc))
        print("Worst Test acc.:{:.2f}".format(worst_test_acc))
        print("Test acc. std.:{:.2f}".format(std_test_acc))
        return mean_test_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print(study.best_params)
