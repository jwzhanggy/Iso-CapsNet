import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
import pickle

NUM_CLASSES = 2
NUM_ROUTING_ITERATIONS = 3

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channel_num, out_channel_num, kernel_size=None, num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channel_num, out_channel_num))
        else:
            self.capsules = Isomorphic_Feature_Extraction(kernel_size, out_channel_num, with_orientation=True)

    def squash(self, tensor, dim=-1):
        #print(tensor)
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        #print(squared_norm)
        scale = squared_norm / (1 + squared_norm)
        #print('scale', scale)
        #print(scale * tensor / torch.sqrt(squared_norm))
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            features = self.capsules(x)
            #print(features)
            outputs = self.squash(features)
            #print(outputs)

        return outputs

class IsoCapsuleNet(nn.Module):
    hyper_parameters = None

    def __init__(self, hyper_parameters):
        self.hyper_parameters = hyper_parameters
        super(IsoCapsuleNet, self).__init__()
        self.primary_capsules = CapsuleLayer(num_capsules=1, num_route_nodes=-1, in_channel_num=1, out_channel_num=1, kernel_size=self.hyper_parameters['k'])
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=self.hyper_parameters['nfeature'], in_channel_num=1*self.hyper_parameters['k']**2,
                                           out_channel_num=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.hyper_parameters['num_node']**2),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        #print(x)
        x = self.primary_capsules(x)
        x[x==0.0]=-1.0
        #print(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        #print(x)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        #classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))
        return classes, reconstructions

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, graphs, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(graphs) == torch.numel(reconstructions)
        graphs = graphs.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, graphs)
        # f = open('./result/HIV_fMRI/intermediate_results', 'wb')
        # result = {'graph': graphs, 'reconstructions': reconstructions, 'y': labels}
        # pickle.dump(result, f)
        # f.close()
        return (margin_loss + 0.0005 * reconstruction_loss) / graphs.size(0)


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math
import numpy as np
from itertools import permutations

from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment

import random


class Isomorphic_Feature_Extraction(nn.Module):
    """docstring for ClassName"""

    def __init__(self, k, c, with_orientation=False):
        super(Isomorphic_Feature_Extraction, self).__init__()
        self.k = k
        self.c = c
        self.with_orientation = with_orientation
        self.fac_k = math.factorial(k)
        self.P = self.get_all_P(self.k)

        self.kernel1 = nn.Parameter(torch.randn(c, k, k))
        self.kernel2 = nn.Parameter(torch.randn(c, k, k))

        self.maxpoolk = nn.MaxPool2d((1, self.fac_k), stride=(1, self.fac_k), return_indices=True)

    def forward(self, x):
        layer1 = self.IsoLayer(x, self.kernel1)
        B, n_subgraph, c, k_square = layer1.size()
        out = layer1.view(B, n_subgraph, c*k_square)
        #print(self.kernel1)
        #out = F.softmax(out, dim=-1)
        return out

    def IsoLayer(self, x, kernel):
        x = self.get_all_subgraphs(x, self.k)
        B, n_subgraph, k, k = x.size()
        x = x.view(-1, 1, self.k, self.k)
        tmp = x - torch.matmul(torch.matmul(self.P, kernel.view(self.c, 1, self.k, self.k)),
                               torch.transpose(self.P, 2, 1)).view(-1, self.k,
                                                                   self.k)  # [B*n_subgraph, 1, k, k] - [c*k!, k, k]
        raw_features = -1 * torch.norm(tmp, p='fro', dim=(-2, -1)) ** 2
        raw_features = raw_features.view(B, n_subgraph, self.c, self.fac_k)

        feature_P, indices = self.maxpoolk(raw_features)
        feature_P = -1*feature_P.view(B, -1, self.c)
        indices = indices.view(B, -1, self.c)
        #print('feature_P', feature_P.shape, feature_P)

        if self.with_orientation:
            capsule = self.P[indices].view(B, -1, self.c, self.k ** 2)
            capsule = capsule * feature_P.unsqueeze(3)
            return capsule
        else:
            return feature_P

    # get all possible P (slow algo)
    def get_all_P(self, k):
        n_P = np.math.factorial(k)
        P_collection = np.zeros([n_P, k, k])
        perms = permutations(range(k), k)

        count = 0
        for p in perms:
            for i in range(len(p)):
                P_collection[count, i, p[i]] = 1
            count += 1
        Ps = torch.from_numpy(np.array(P_collection)).requires_grad_(False)
        Ps = Ps.type(torch.FloatTensor)

        return Ps

    def get_all_subgraphs(self, X, k):
        X = X.detach().squeeze()
        (batch_size, n_H_prev, n_W_prev) = X.size()
        n_H = n_H_prev - k + 1
        n_W = n_W_prev - k + 1
        subgraphs = []
        for h in range(n_H):
            for w in range(n_W):
                x_slice = X[:, h:h + k, w:w + k]
                subgraphs.append(x_slice)
        S = torch.stack(subgraphs, dim=1)
        return S


class Classification_Component(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, nclass):
        super(Classification_Component, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, nclass)

    def forward(self, features):
        h1 = F.relu(self.fc1(features))
        h2 = F.relu(self.fc2(h1))
        pred = F.log_softmax(self.fc3(h2), dim=1)
        return pred







