import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import time
## r=1
from torch_geometric.nn import global_max_pool

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch.nn import Sequential as Seq, Dropout, Linear as Lin

try:
    import graphOps as GO
    from src.graphOps import getConnectivity
    from mpl_toolkits.mplot3d import Axes3D
    from src.utils import saveMesh, h_swish
    from src.inits import glorot, identityInit

except:
    import graphOps as GO
    from graphOps import getConnectivity
    from mpl_toolkits.mplot3d import Axes3D
    # from utils import saveMesh, h_swish
    import utils
    import inits
    from inits import glorot, identityInit
    # import inits

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCN2Conv
from torch_scatter import scatter_add

# Function for 2D convolution operation
def conv2(X, Kernel):
    """
    Perform 2D convolution operation on input tensor X using the specified kernel.

    Parameters:
    - X (torch.Tensor): Input tensor.
    - Kernel (torch.Tensor): Convolution kernel.

    Returns:
    - torch.Tensor: Result of the 2D convolution operation.
    """
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


# Function for 1D convolution operation
def conv1(X, Kernel):
    """
    Perform 1D convolution operation on input tensor X using the specified kernel.

    Parameters:
    - X (torch.Tensor): Input tensor.
    - Kernel (torch.Tensor): Convolution kernel.

    Returns:
    - torch.Tensor: Result of the 1D convolution operation.
    """
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


# Function for 1D transposed convolution operation
def conv1T(X, Kernel):
    """
    Perform 1D transposed convolution operation on input tensor X using the specified kernel.

    Parameters:
    - X (torch.Tensor): Input tensor.
    - Kernel (torch.Tensor): Transposed convolution kernel.

    Returns:
    - torch.Tensor: Result of the 1D transposed convolution operation.
    """
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

# Function for 2D transposed convolution operation
def conv2T(X, Kernel):
    """
    Perform 2D transposed convolution operation on input tensor X using the specified kernel.

    Parameters:
    - X (torch.Tensor): Input tensor.
    - Kernel (torch.Tensor): Transposed convolution kernel.

    Returns:
    - torch.Tensor: Result of the 2D transposed convolution operation.
    """
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cuda:0'



# Function for total variation (TV) normalization
def tv_norm(X, eps=1e-3):
    """
    Perform total variation (TV) normalization on input tensor X.

    Parameters:
    - X (torch.Tensor): Input tensor.
    - eps (float): Small constant to avoid division by zero.

    Returns:
    - torch.Tensor: Result of TV normalization.
    """
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


# Function for computing differences along the X-axis
def diffX(X):
    """
    Compute differences along the X-axis of the input tensor X.

    Parameters:
    - X (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Result of differencing along the X-axis.
    """
    X = X.squeeze()
    return X[:, 1:] - X[:, :-1]


# Function for computing transposed differences along the X-axis
def diffXT(X):
    """
    Compute transposed differences along the X-axis of the input tensor X.

    Parameters:
    - X (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Result of transposed differencing along the X-axis.
    """
    X = X.squeeze()
    D = X[:, :-1] - X[:, 1:]
    d0 = -X[:, 0].unsqueeze(1)
    d1 = X[:, -1].unsqueeze(1)
    D = torch.cat([d0, D, d1], dim=1)
    return D



# Function for a double layer operation
def doubleLayer(x, K1, K2):
    """
    Perform a double layer operation on input tensor x using two specified convolution kernels.

    Parameters:
    - x (torch.Tensor): Input tensor.
    - K1 (torch.Tensor): First convolution kernel.
    - K2 (torch.Tensor): Second convolution kernel.

    Returns:
    - torch.Tensor: Result of the double layer operation.
    """
    x = F.conv1d(x, K1.unsqueeze(-1))
    x = F.layer_norm(x, x.shape)
    x = torch.relu(x)
    x = F.conv1d(x, K2.unsqueeze(-1))
    return x


###################################################################################pdegcn

    
def MLP(channels, batch_norm=True):
    """
    Create a Multi-Layer Perceptron (MLP) neural network.

    Parameters:
    - channels (list): List of integers representing the number of input and output channels for each layer.
    - batch_norm (bool): Flag indicating whether to include batch normalization.

    Returns:
    - torch.nn.Sequential: MLP model.
    """
    # Define a list comprehension to create a sequence of layers for the MLP
    layers = [
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ]

    # Create a Sequential model using the defined layers
    mlp_model = Seq(*layers)

    return mlp_model



class graphNetwork_nodesOnly(nn.Module):

    def __init__(self, nNin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
                 diffOrder=1, num_output=1024, dropOut=False, modelnet=False, faust=False, GCNII=False,
                 graphUpdate=None, PPI=False, gated=False, realVarlet=False, mixDyamics=False, doubleConv=False,
                 tripleConv=False):
         """
        Graph neural network model for node-level operations.
        """
        super(graphNetwork_nodesOnly, self).__init__()
        self.wave = wave
        self.realVarlet = realVarlet
        if not wave:
            self.heat = True
        else:
            self.heat = False
        self.mixDynamics = mixDyamics
        self.h = h
        self.varlet = varlet
        self.dense = dense
        self.diffOrder = diffOrder
        self.num_output = num_output
        self.graphUpdate = graphUpdate
        self.doubleConv = doubleConv
        self.tripleConv = tripleConv
        self.gated = gated
        self.faust = faust
        self.PPI = PPI
        if dropOut > 0.0:
            self.dropout = dropOut
        else:
            self.dropout = False
        self.nlayers = nlayer
        stdv = 1e-2
        stdvp = 1e-2
        if self.faust or self.PPI:
            stdv = 1e-1
            stdvp = 1e-1
            stdv = 1e-2
            stdvp = 1e-2
        
        # Add learnable parameter D to edges
        self.omega = nn.Parameter(torch.ones(13264)) # Cora
        # self.omega = nn.Parameter(torch.ones(12431)) # citeseer
        # self.omega = nn.Parameter(torch.ones(108365)) # constant Pub
        # self.omega = nn.Parameter(torch.ones(41328)) # Faust
        # self.omega = nn.Parameter(torch.ones(nlayer, 12431))  # m x m diag
        # self.omega = nn.Parameter(torch.ones(nlayer, nopen))
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.convs1x1 = nn.Parameter(torch.randn(nlayer, nopen, nopen) * stdv)
        self.modelnet = modelnet

        if self.modelnet:
            self.KNclose = nn.Parameter(torch.randn(1024, num_output) * stdv)  # num_output on left size
        elif not self.faust:
            self.KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)  # num_output on left size
        else:
            self.KNclose = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        if varlet:
            Nfeatures = 1 * nopen
        else:
            Nfeatures = 1 * nopen

        self.KN1 = nn.Parameter(torch.rand(nlayer, Nfeatures, nhid) * stdvp)
        rrnd = torch.rand(nlayer, Nfeatures, nhid) * (1e-3)

        self.KN1 = nn.Parameter(identityInit(self.KN1) + rrnd)

        if self.realVarlet:
            self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, 2 * Nfeatures) * stdvp)
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, 2 * Nfeatures) * stdvp)

        if self.mixDynamics:
            self.alpha = nn.Parameter(-0 * torch.ones(1, 1))
            self.beta = nn.Parameter(-0 * torch.ones(1, 1))

        self.KN2 = nn.Parameter(torch.rand(nlayer, nhid, 1 * nhid) * stdvp)
        self.KN2 = nn.Parameter(identityInit(self.KN2))

        if self.tripleConv:
            self.KN3 = nn.Parameter(torch.rand(nlayer, nopen, 1 * nhid) * stdvp)
            self.KN3 = nn.Parameter(identityInit(self.KN3))

        if self.faust:
            self.lin1 = torch.nn.Linear(nopen, nopen)
            self.lin2 = torch.nn.Linear(nopen, num_output)

        self.modelnet = modelnet

        self.PPI = PPI
        if self.modelnet:
            self.mlp = Seq(
                MLP([64, 128]), Dropout(0.5), MLP([128, 64]), Dropout(0.5),
                Lin(64, 10))

    def reset_parameters(self):
        """
        Reset parameters of the model using Glorot initialization.
        """
        glorot(self.K1Nopen)
        glorot(self.K2Nopen)
        glorot(self.KNclose)
        if self.realVarlet:
            glorot(self.KE1)
        if self.modelnet:
            glorot(self.mlp)

    def edgeConv(self, xe, K, groups=1):
        """
        Perform edge convolution on the input tensor xe using the specified kernel K.

        Parameters:
        - xe (torch.Tensor): Input tensor.
        - K (torch.Tensor): Convolution kernel.
        - groups (int): Number of groups for grouped convolution.

        Returns:
        - torch.Tensor: Result of the edge convolution.
        """
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1), groups=groups)
            else:
                xe = conv2(xe, K, groups=groups)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1), groups=groups)
            else:
                xe = conv1(xe, K, groups=groups)
        return xe

    def singleLayer(self, x, K, relu=True, norm=False, groups=1, openclose=False):
        """
        Perform a single-layer operation on the input tensor x using the specified kernel K.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - K (torch.Tensor): Convolution kernel.
        - relu (bool): Apply ReLU activation if True, apply tanh if False.
        - norm (bool): Apply instance normalization if True.
        - groups (int): Number of groups for grouped convolution.
        - openclose (bool): Use open-close operation if True, close-open operation if False.

        Returns:
        - torch.Tensor: Result of the single-layer operation.
        """
        if openclose:  # if K.shape[0] != K.shape[1]:
            x = self.edgeConv(x, K, groups=groups)
            if norm:
                x = F.instance_norm(x)
            if relu:
                x = F.relu(x)
            else:
                x = F.tanh(x)
        if not openclose:  # if K.shape[0] == K.shape[1]:
            x = self.edgeConv(x, K, groups=groups)
            if not relu:
                x = F.tanh(x)
            else:
                x = F.relu(x)
            if norm:
                beta = torch.norm(x)
                x = beta * tv_norm(x)
            x = self.edgeConv(x, K.t(), groups=groups)
        return x

    def finalDoubleLayer(self, x, K1, K2):
        """
        Perform a final double-layer operation on the input tensor x using two specified kernels.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - K1 (torch.Tensor): First convolution kernel.
        - K2 (torch.Tensor): Second convolution kernel.

        Returns:
        - torch.Tensor: Result of the final double-layer operation.
        """
        x = F.tanh(x)
        x = self.edgeConv(x, K1)
        x = F.tanh(x)
        x = self.edgeConv(x, K2)
        x = F.tanh(x)
        x = self.edgeConv(x, K2.t())
        x = F.tanh(x)
        x = self.edgeConv(x, K1.t())
        x = F.tanh(x)
        return x

    def savePropagationImage(self, xn, Graph, i=0, minv=None, maxv=None):
        """
        Save and display an image representing the propagation of features in the graph.

        Parameters:
        - xn (torch.Tensor): Input tensor.
        - Graph: Graph object.
        - i (int): Index for naming the saved image.
        - minv (float): Minimum value for the color map.
        - maxv (float): Maximum value for the color map.
        """
        plt.figure()
        img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        if (maxv is not None) and (minv is not None):
            plt.imshow(img, vmax=maxv, vmin=minv)
        else:
            plt.imshow(img)

        plt.colorbar()
        plt.show()
        plt.savefig('plots/layer' + str(i) + '.jpg')

        plt.close()

    def updateGraph(self, Graph, features=None):
        """
        Update the graph based on features.

        Parameters:
        - Graph: Graph object.
        - features (torch.Tensor): Input features.

        Returns:
        - Graph: Updated graph.
        """
        # If features are given - update graph according to feaure space l2 distance
        N = Graph.nnodes
        I = Graph.iInd
        J = Graph.jInd
        edge_index = torch.cat([I.unsqueeze(0), J.unsqueeze(0)], dim=0)
        if features is not None:
            features = features.squeeze()
            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)
            D = D / D.std()
            D = torch.exp(-2 * D)
            w = D[I, J]
            Graph = GO.graph(I, J, N, W=w, pos=None, faces=None)

        else:
            [edge_index, edge_weights] = gcn_norm(edge_index)  # Pre-process GCN normalization.
            I = edge_index[0, :]
            J = edge_index[1, :]
            # deg = self.getDegreeMat(Graph)
            Graph = GO.graph(I, J, N, W=edge_weights, pos=None, faces=None)

        return Graph, edge_index

    def forward(self, xn, Graph, data=None, xe=None):
        """
        Forward pass of the graph neural network.

        Parameters:
        - xn (torch.Tensor): Input tensor.
        - Graph: Graph object.
        - data: Additional data (not used in the provided code).
        - xe (torch.Tensor): Edge tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer
        if not self.faust:
            [Graph, edge_index] = self.updateGraph(Graph)
        if self.faust:
            xn = torch.cat([xn, Graph.edgeDiv(xe)], dim=1)
        xhist = []
        debug = False

        if debug:
            xnnorm = torch.norm(xn, dim=1)
            vmin = xnnorm.min().detach().numpy()
            vmax = xnnorm.max().detach().numpy()
            saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, -1, vmax=vmax, vmin=vmin)

        if self.realVarlet:
            xe = Graph.nodeGrad(xn)
            if self.dropout:
                xe = F.dropout(xe, p=self.dropout, training=self.training)
            xe = self.singleLayer(xe, self.K2Nopen, relu=True)

        if self.dropout:
            xn = F.dropout(xn, p=self.dropout, training=self.training)

        xn = self.singleLayer(xn, self.K1Nopen, relu=True, openclose=True, norm=False)
        # print("old xn")
        # print(xn.size())
        x0 = xn.clone()
        debug = False
        if debug:
            image = False
            if image:
                plt.figure()
                # print("xn shape:", xn.shape)
                img = xn.clone().detach().squeeze().cpu().numpy().reshape(32, 32)
                minv = img.min()
                maxv = img.max()
                # img = img / img.max()
                plt.imshow(img, vmax=maxv, vmin=minv)
                plt.colorbar()
                plt.show()
                plt.savefig('plots/img_xn_norm_layer_verlet' + str(1) + 'order_nodeDeriv' + str(0) + '.jpg')
                plt.close()
            else:
                saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, 0, vmax=vmax, vmin=vmin)

        xn_old = x0
        # print("xn_old")
        # print(xn_old.size())
        nlayers = self.nlayers
        for i in range(nlayers):
            
            if self.graphUpdate is not None:
                if i % self.graphUpdate == self.graphUpdate - 1:  # update graph

                    Graph, edge_index = self.updateGraph(Graph, features=xn)
                    dxe = Graph.nodeAve(xn)

            bur = 0
            # bur == 0 : wave mixing
            # bur == 1 : first burger model
            # bur == 2 : second burger model
            # bur == 3 : third burger model
            # bur == 4 : diffusive mixing
            # bur == 5 : advection
            # fully / fause run this 
            if not self.realVarlet:
                # fully/faust
                if self.varlet:
                    
                    if bur == 2:
                        bur2X = xn * xn 
                        bur2X = Graph.nodeAve(bur2X)
                        bur2X = self.edgeConv(bur2X, self.KN1[i], groups=1)
                        bur2X = Graph.edgeDiv(bur2X)
                    elif bur == 3:
                        bur3X = self.edgeConv(xn, self.KN1[i], groups=1)
                        bur3x1 = self.edgeConv(xn, self.KN2[i], groups=1)
                        bur3X = bur3x1 * bur3X
                        bur3X = Graph.nodeAve(bur3X)
                        bur3X = Graph.edgeDiv(bur3X)
                    elif bur == 1:
                        # burger:
                        aveX = Graph.nodeAve(xn)
                        burX = self.edgeConv(aveX, self.KN1[i], groups=1)
                        burX = burX * burX
                        burX = F.tanh(burX)
                        burX = Graph.edgeDiv(burX)
                    elif bur == 5:
                        # A^T u
                        aveX = Graph.nodeAve(xn)
                        # G^T A^T u
                        dxn = Graph.edgeDiv(aveX)

                        if self.dropout:
                            if self.varlet:
                                aveX = F.dropout(aveX, p=self.dropout, training=self.training)
                        # sigma()
                        # sigma(D A^T u K)
                        aveX = (self.singleLayer(aveX, self.KN1[i], norm=False, relu=True, groups=1))  # KN1

                    else:
                        # G u
                        gradX = Graph.nodeGrad(xn)
                        gradX = F.relu(self.omega[0].view(1, 1, -1)) * (gradX)
                        # A^T u
                        aveX = Graph.nodeAve(xn)
                        # D A^T u
                        aveX = F.relu(self.omega[0].view(1, 1, -1)) * (aveX)
                        # G^T D A^T u
                        # dxn = Graph.edgeDiv(aveX)
                        dxn = aveX
                        if self.dropout:
                            if self.varlet:
                                aveX = F.dropout(aveX, p=self.dropout, training=self.training)
                                gradX = F.dropout(gradX, p=self.dropout, training=self.training)
                        # sigma()
                        # sigma(D A^T u K)
                        aveX = (self.singleLayer(aveX, self.KN1[i], norm=False, relu=True, groups=1))  # KN1
                        gradX = (self.singleLayer(gradX, self.KN2[i], norm=False, relu=True, groups=1))  # KN2
                        aveX = F.relu(self.omega[0].view(1, 1, -1)) * (aveX)
                        gradX = F.relu(self.omega[0].view(1, 1, -1)) * (gradX)
                        # print("sec dxn")
                        # print(dxn.size())
                        # G^T sigma(D A^T u K)
                        gradX = Graph.edgeDiv(gradX)
                        aveX = Graph.edgeDiv(aveX)

            
                if self.mixDynamics:
                    tmp_xn = xn.clone()

                    if bur == 1: # Burger no.1
                        h = self.h
                        xn = -h/2 * burX + xn
                    elif bur == 2: # Burger no.2
                        h = self.h
                        xn = -h/2 * bur2X + xn
                    elif bur == 3: # Burger no.3
                        h = self.h
                        xn = -h * bur3X + xn
                    elif bur == 0: # hyperbolic mixing
                        beta = F.sigmoid(self.alpha)
                        alpha = 1 - beta
                
                        alpha = alpha / self.h
                        beta = beta / (self.h ** 2)
                        alpha = alpha.to(device)
                        beta = beta.to(device)

                        xn = xn.to(device)
                        xn_old = xn_old.to(device)
                        aveX = aveX.to(device)
                        gradX = gradX.to(device)    
                        alpha1 = F.sigmoid(self.alpha).to(device)
                        xn1 = 2 * beta * xn 
                        xn2 = - beta * xn_old
                        xn3 = alpha * xn
                        xn4 = - 1 / self.h *alpha1 * aveX
                        xn5 = - (1-alpha1) * gradX
                        xn6 = beta + alpha
                        xn = (xn1 + xn2 + xn3 + xn4 + xn5) / xn6
                        # xn = (2 * beta * xn - beta * xn_old + alpha * xn - 1 / self.h *alpha1 * aveX - (1-alpha1) * gradX) / (beta + alpha)
                    elif bur == 5: # advection
                        xn = - self.h * aveX + xn
                    else: # diffusive mixing
                        beta = F.sigmoid(self.alpha)
                        alpha = 1 - beta
                
                        alpha = alpha * (self.h)
                        beta = beta *self.h 

                        beta = beta.to(device)
                        alpha = alpha.to(device)
                        aveX = aveX.to(device)
                        gradX = gradX.to(device)
                        xn = xn.to(device)

                        xn = -beta * aveX -alpha * gradX + xn
                    

                    xn_old = tmp_xn

                elif self.wave:
                    tmp_xn = xn.clone()
                    xn = 2 * xn - xn_old - (self.h ** 2) * dxn
                    xn_old = tmp_xn
                else:
                    tmp = xn.clone()
                    xn = (xn - self.h * dxn)
                    xn_old = tmp

                if self.modelnet:
                    xhist.append(xn)

            if debug:
                if image:
                    self.savePropagationImage(xn, Graph, i + 1, minv=minv, maxv=maxv)
                else:
                    saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, i + 1, vmax=vmax, vmin=vmin)

        xn = F.dropout(xn, p=self.dropout, training=self.training)
        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        # print('alpha:')
        # print(beta)
        xn = xn.squeeze().t()
        if self.modelnet:
            out = global_max_pool(xn, data.batch)
            out = self.mlp(out)
            return F.log_softmax(out, dim=-1)

        if self.faust:
            self.alpha = nn.Parameter(0.5 * torch.ones(1, 1))
            x = F.relu(self.lin1(xn))
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1), F.sigmoid(self.alpha)

        if self.PPI:
            return xn, Graph

        ## Otherwise its citation graph node classification:
        return F.log_softmax(xn, dim=1), Graph
