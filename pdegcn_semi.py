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

import sys

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties('cuda:0'))

import graphOps as GO
# from src import processContacts as prc
import utils
import graphNet as GN

num_layers = [2,4,8,16,32,64]
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties('cuda:0'))

# Setup the network and its parameters
for nlayers in num_layers:
    torch.cuda.synchronize()
    print("Doing experiment for ", nlayers, " layers!", flush=True)
    torch.cuda.synchronize()

    def objective(trial):
        dataset = 'PubMed'
        if dataset == 'Cora':
            nNin = 1433
        elif dataset == 'CiteSeer':
            nNin = 3703
        elif dataset == 'PubMed':
            nNin = 500
        nEin = 1
        n_channels = 256  # trial.suggest_categorical('n_channels', [64, 128, 256])
        nopen = n_channels
        nhid = n_channels
        nNclose = n_channels
        n_layers = nlayers
        print("DATA SET IS:", dataset)
        # h = 1 / n_layers
        # h = trial.suggest_discrete_uniform('h', 1 / (n_layers), 3, q=1 / (n_layers))
        # h = trial.suggest_discrete_uniform('h', 0.1, 3, q=0.1)
        h =0.7
        batchSize = 32

        path = ''
        transform = T.Compose([T.NormalizeFeatures()])
        dataset = Planetoid(path, dataset, transform=transform)
        data = dataset[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        # dropout = trial.suggest_discrete_uniform('dropout', 0.6, 0.8, q=0.1)
        # lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        # lrGCN = trial.suggest_float("lrGCN", 1e-5, 1e-3, log=True)
        # wd = trial.suggest_float("wd", 5e-6, 1e-3, log=True)
        # lr_alpha = trial.suggest_float("lr_alpha", 1e-5, 1e-2, log=True)
        
        # CiteSeer
        # dropout = 0.7
        # lr = 7e-2
        # lrGCN = 2e-6
        # wd = 0.003

        # Cora
        # dropout = 0.7
        # lr = 7e-2
        # lrGCN = 5e-5
        # wd = 5e-4

        # PubMed
        dropout = 0.7
        lr = 3e-2
        lrGCN = 3e-5
        wd = 1e-4
        model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, n_layers, h=h, dense=False, varlet=True,
                                          wave=False,
                                          diffOrder=1, num_output=dataset.num_classes, dropOut=dropout, gated=False,
                                          realVarlet=False, mixDyamics=True, doubleConv=False, tripleConv=False)

        model.reset_parameters()
        model.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.KN1, lr=lrGCN, weight_decay=0),
            dict(params=model.KN2, lr=lrGCN, weight_decay=0),
            dict(params=model.K1Nopen, weight_decay=wd),
            dict(params=model.KNclose, weight_decay=wd),

        ], lr=lr)

        def train():
            model.train()
            optimizer.zero_grad()

            I = data.edge_index[0, :]
            J = data.edge_index[1, :]
            N = data.y.shape[0]

            features = data.x.squeeze().t()

            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)
            D = D / D.std()
            D = torch.exp(-2 * D)

            w = D[I, J] #tmp. replaced inside the net for gcn norm
            G = GO.graph(I, J, N, W=w, pos=None, faces=None)
            G = G.to(device)
            xn = data.x.t().unsqueeze(0)
            [out, G] = model(xn, G)
            [valmax, argmax] = torch.max(out, dim=1)
            g = G.nodeGrad(out.t().unsqueeze(0))
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            return float(loss)

        @torch.no_grad()
        def test():
            model.eval()
            I = data.edge_index[0, :]
            J = data.edge_index[1, :]
            N = data.y.shape[0]
            features = data.x.squeeze().t()
            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)

            D = D / D.std()
            D = torch.exp(-2 * D)
            w = D[I, J]
            G = GO.graph(I, J, N, W=w, pos=None, faces=None)
            G = G.to(device)
            xn = data.x.t().unsqueeze(0)
            [out, G] = model(xn, G)
            pred, accs = out.argmax(dim=-1), []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
            return accs
        start = time.time()
        best_val_acc = test_acc = 0

        for epoch in range(2000):
            
            loss = train()
            train_acc, val_acc, tmp_test_acc = test()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                # print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
                #       f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
                #       f'Final Test: {test_acc:.4f}')
        end = time.time()
        
        print("running time:{:.2f}".format(end - start))
        print("test acc.:{:.2f}".format(test_acc *100))
        return test_acc *100


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)
