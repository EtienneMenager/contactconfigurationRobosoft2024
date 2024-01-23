# -*- coding: utf-8 -*-
"""Create the agent model.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

from sac import MLPNetwork

class Selector(nn.Module):
    def __init__(self, d_input, d_q, learning_rate=1e-4, hidden_size = 256):
        super(Selector, self).__init__()

        self.Fk = MLPNetwork(d_input, d_q, hidden_size=hidden_size) #nn.Linear(d_input, d_q)
        self.index_to_qt = dict()
        self.softmax = nn.Softmax(dim=0)
        self.temperature = np.power(d_q, 0.5)
        self.updateOptimizer(learning_rate)

    def updateOptimizer(self,learning_rate):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def notify(self, index, qt, remove = False):
        if remove:
            self.index_to_qt.pop(index)
        else:
            self.index_to_qt.update({index:qt.view(1, -1)})

    def forward(self, x, Q):
        if len(x.size())==1:
            x = x.view(1, -1)
        K = self.Fk(x)
        attn = torch.matmul(Q, K.transpose(1,0))
        attn = attn/self.temperature
        attn = self.softmax(attn)

        if x.size()[0]==1:
            attn = attn.view(1, -1)

        return attn

    def findBest(self, x):
        Q = torch.cat(list(self.index_to_qt.values()), dim = 0)
        attn = self.forward(x, Q)
        id = attn.argmax(dim = 1)

        return attn, id

    def save_model(self, file = './Results', name = 'selector_latest'):
        path_model = file + '/' + name +'.pth'
        os.makedirs(file , exist_ok=True)
        torch.save(self.state_dict(), path_model)

        print(">>    Model saved at {}".format(path_model))

    def load_model(self,file = './Results', name = 'selector_latest'):
        path_model = file + '/' + name +'.pth'
        self.load_state_dict(torch.load(path_model), strict=False)

        print(">>    Model loaded at {}".format(path_model))


class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()

    def nodesEvaluation(self, information):
        return information

    def save_model(self, file = './Results', name = 'evaluator_latest'):
        path_model = file + '/' + name +'.pth'
        os.makedirs(file , exist_ok=True)
        torch.save(self.state_dict(), path_model)
        print(">>    Model saved at {}".format(path_model))

    def load_model(self,file = './Results', name = 'evaluator_latest'):
        path_model = file + '/' + name +'.pth'
        self.load_state_dict(torch.load(path_model), strict=False)
        print(">>    Model loaded at {}".format(path_model))


class Formatter(nn.Module):
    def __init__(self):
        super(Formatter, self).__init__()

    def createRepresentation(self, x, g):
        pass

    def forward(self, x, g):
        output = self.createRepresentation(x, g)
        return output

    def save_model(self, file = './Results', name = 'formatter_latest'):
        path_model = file + '/' + name +'.pth'
        os.makedirs(file , exist_ok=True)
        torch.save(self.state_dict(), path_model)
        print(">>    Model saved at {}".format(path_model))

    def load_model(self,file = './Results', name = 'formatter_latest'):
        path_model = file + '/' + name +'.pth'
        self.load_state_dict(torch.load(path_model), strict=False)
        print(">>    Model loaded at {}".format(path_model))


##########################



if __name__ == '__main__':
    d_input, d_q, d_v = 4, 8, 8
    d_h, d_hidden = 8, 8
    action_dim = 8

    selector = Selector(d_input, d_q)
    print(selector)

    x = torch.randn(1, d_input)
    q0 = torch.randn(1, d_q)
    q1 = torch.randn(1, d_q)
    q2 = torch.randn(1, d_q)

    selector.notify(0, q0)
    selector.notify(1, q1)
    selector.notify(2, q2)

    print(">>  Notification:", selector.index_to_qt)
    print(">>  x:", x)

    Q = torch.cat([q0, q1, q2], dim = 0)

    print(">>  Q:", Q)

    attn, V = selector(x, Q)
    print(">>  Attention and value:", attn, V)

    x = torch.randn(1000, d_input)
    attn, V = selector(x, Q)
    print(">>  MULTIPLE Attention and value:", attn, V)
    print(">>  Index max:", attn.argmax(dim=0))
