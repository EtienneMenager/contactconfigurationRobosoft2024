# -*- coding: utf-8 -*-
"""Create the lms model.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

from classState import ClassState
from agent import Selector
import torch.nn.functional as F
from visualization import Viewer
import torch
import torch.nn as nn
import os
import numpy as np
import json
import random
from tqdm import tqdm

from itertools import groupby
import operator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque, namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))

Batch = namedtuple('Batch', ('batch_state', 'batch_action', 'batch_nextstate'))


class Buffer:
    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))

    def push(self, batch):
        """ Saves a transition """
        self._memory.append(batch)

    def sample(self, batch_size: int):
        batch = random.sample(self._memory, batch_size)
        return Batch(*zip(*batch))

    def get(self, start_idx: int, end_idx: int):
        batch = list(itertools.islice(self._memory, start_idx, end_idx))
        return Batch(*zip(*batch))

    def get_all(self):
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

class LMS(nn.Module):
    def __init__(self, formatter, selector, config = None):
        super(LMS, self).__init__()

        if config is None:
            print(">>  Please load a model or reinit with a config.")
        else:
            self.reinit(config, formatter, selector)

        self.criterionHandwritten = nn.CrossEntropyLoss()

    def reinit(self, config, formatter, selector):
        self.config = config
        self.d_hidden = config["d_hidden"]
        self.d_output = config["d_output"]
        self.d_input = config["d_input"]
        self.batchsize = config["batchsize"]
        self.gamma = config["gamma"]
        self.pre_learning = config["pre_learning"]
        self.reward_external = config["reward_external"]
        self.use_LSTM_encoder = config["use_LSTM_encoder"]

        self.action_dim = config["action_dim"]
        self.action_lim = config["action_lim"]
        self.continous = config["continous"]
        self.type_attention_evaluator = config["type_attention_evaluator"]

        [use_evaluator, use_external_agent, use_inside_agent] = config["use_agents"]

        self.selector = selector
        self.formatter = formatter
        self.Buffer = Buffer()

        self.viewer = Viewer()

        self.index_to_nodes = dict()
        if "nodes" in config:
            n_state = len(config["nodes"])
            for i in range(n_state):
                c = ClassState(self.d_input, self.d_output, self.d_hidden, self.action_dim, self.viewer, n_state, use_LSTM_encoder = self.use_LSTM_encoder, action_lim=self.action_lim,  batchsize = self.batchsize, gamma = self.gamma, reward_external=self.reward_external, use_evaluator = use_evaluator, use_external_agent = use_external_agent, use_inside_agent = use_inside_agent, pre_learning = self.pre_learning, name = "ClassState"+str(i), continuous= self.continous, type_attention_evaluator = self.type_attention_evaluator)
                c.notifyAgent(self.selector)
                self.index_to_nodes.update({c.index:c})

            for i, n in enumerate(self.index_to_nodes.values()):
                v = config["nodes"][i]
                for elt in v:
                    n.addNeighbours([self.index_to_nodes[elt]])
                n.setPsucc(config["p"][i])

            pagg = []
            for n in self.index_to_nodes.values():
                pagg.append(n.computePagg(k=0))
            for i, n in enumerate(self.index_to_nodes.values()):
                n.setPagg(pagg[i])

        self.old_node = config["init_active"]
        self.viewer.updateColorG(self.old_node, self.old_node)
        self.updateModuleList()

        for node in self.modules_list:
            node.create_agents(self)

    def updateModuleList(self):
        m = list(self.index_to_nodes.values())
        i = list(self.index_to_nodes.keys())

        self.modules_list = nn.ModuleList(m)
        self.index_to_modules = dict(zip(i, list(range(len(i)))))

    def updateP(self, p, k_max = 0):
        for i, n in enumerate(self.modules_list):
            n.setPsucc(p[i])

        for k in range(k_max+1):
            new_values = []
            for n in self.modules_list:
                new_values.append(n.computePagg(k))
            for i, n in enumerate(self.modules_list):
                n.setPagg(new_values[i])

    def setP(self, info, k_max, deterministic = False):
        p = self.evaluator.nodesEvaluation(info, deterministic = deterministic)
        self.updateP(p, k_max = k_max)

    def deleteNode(self, index):
        node = self.index_to_nodes[index]
        succ = node.delete(self.selector)
        for s in succ:
            n = self.index_to_nodes[s]
            n.delNeighbour(index)
        self.index_to_nodes.pop(index)
        self.updateModuleList()

    def forward(self, x):
        print(">>>   Please use one step learning to realise learning.")
        exit(1)

    def f_dyn(self, x, h, c, idx):
        if len(x.size())==1:
            x = x.view(1, -1)

        idx_mod = self.index_to_modules[idx]
        module = self.modules_list[idx_mod]
        h_new, c_new, q_new = module(x,h, c)
        self.updateActive(idx)

        return h_new, c_new, q_new

    def _learningSelector(self, X, Y):
        batch_size = X.size()[0]

        for p in self.selector.parameters():
            p.requires_grad = False
        for n in self.modules_list:
            for p in n.parameters():
                p.requires_grad = True


        h, c, q = [], [], []
        for n in self.modules_list:
            _h = n.ht.detach()
            h.append(n.ht.detach())
            c.append(n.ct.detach())
            q.append(n.compute_qt())

        loss_d = torch.zeros(1, requires_grad=True)
        for i, x in enumerate(X):
            idx = int(Y[i])
            idx_mod = self.index_to_modules[idx]

            Q = torch.cat(q, dim = 0)
            attn = self.selector(x, Q)

            h[idx_mod], c[idx_mod], q[idx_mod] = self.f_dyn(x, h[idx_mod], c[idx_mod], idx)
            loss_d = loss_d + self.criterionHandwritten(attn, torch.tensor([idx_mod]))/batch_size

        for i, n in enumerate(self.modules_list):
            n.optimizer.zero_grad()
        loss_d.backward(retain_graph=True)
        for i, n in enumerate(self.modules_list):
            n.optimizer.step()

        for i, n in enumerate(self.modules_list):
            n.updateMemory(h[i], c[i], q[i])
            n.notifyAgent(self.selector)

        h, c, q = [], [], []
        for n in self.modules_list:
            h.append(n.ht.detach())
            c.append(n.ct.detach())
            q.append(n.qt.detach())
        Q = torch.cat(q, dim = 0)

        for p in self.selector.parameters():
            p.requires_grad = True
        for n in self.modules_list:
            for p in n.parameters():
                p.requires_grad = False

        loss_c = torch.zeros(1, requires_grad=True)
        for i, x in enumerate(X):
            idx = int(Y[i])
            idx_mod = self.index_to_modules[idx]
            attn = self.selector(x, Q)
            loss_c = loss_c+self.criterionHandwritten(attn, torch.tensor([idx_mod]))/(batch_size)

        self.selector.optimizer.zero_grad()
        loss_c.backward(retain_graph=True)
        self.selector.optimizer.step()

        for n in self.modules_list:
            for p in n.parameters():
                p.requires_grad = True

        return loss_d.item(), loss_c.item()

    def one_step_learningHandWritten(self, X, Y):
        return self._learningSelector(X, Y)

    def evalHandwritten(self, X, Y):
        q = []
        for n in self.modules_list:
            q.append(n.qt.detach())
        Q = torch.cat(q, dim = 0)

        attn = self.selector(X, Q)
        idx = attn.argmax(dim=0)

        correct, non_correct = 0, 0
        for i, id in tqdm(enumerate(idx)):
            if id == int(Y[i]):
                correct+=1
            else:
                non_correct+=1

        return correct, non_correct

    def updateActive(self, id):
        self.viewer.updateColorG(id, self.old_node)
        self.old_node = id

    def findActive(self, x):
        _, id = self.selector.findBest(x)
        return self.modules_list[id[0]].index, self.modules_list[id[0]]

    def optimize_agents_inside(self, n_update):
        for mod in self.modules_list:
            if mod.inside_agent.replay_pool.__len__()> mod.inside_agent.batchsize:
                mod.inside_agent.optimize(n_update, self)

    def optimize_agents_external(self, n_update):
        for mod in self.modules_list:
            if mod.external_agent.replay_pool.__len__()> mod.external_agent.batchsize:
                mod.external_agent.optimize(n_update)

    def optimize_evaluator(self, n_update):
        for mod in self.modules_list:
            if mod.evaluator.replay_pool.__len__()> mod.evaluator.batchsize:
                mod.evaluator.optimize(n_update, self)

    def save_model(self, file = './Results', name = 'latest'):
        path_config = file + "/config_"+ name + ".txt"
        os.makedirs(file , exist_ok=True)

        m = list(self.index_to_nodes.values())

        p = []
        n = []

        for elt in m:
            next = []
            for succ in list(self.viewer.G.G.successors(elt.index)):
                next.append(self.index_to_modules[succ])
            n.append(next)
            p.append(str(elt.getPsucc()))

        self.config.update({"nodes": n, "p": p, "init_active": self.old_node})

        for i, elt in enumerate(m):
            elt.save_model(file = file, name = 'classState'+str(i)+'_'+name)
        self.selector.save_model(file = file, name = 'selector_'+name)
        #self.formatter.save_model(file = file, name = 'formatter_'+name)

        with open(path_config, 'w') as outfile:
            json.dump(self.config, outfile)

        print(">>    Model saved in {}".format(file))
        print(">>    Config saved at {}".format(path_config))

    def load_model(self, formatter, selector, file = './Results', name = 'latest', load_config = False):
        if load_config:
            path_config = file + "/config_"+ name + ".txt"
            with open(path_config, 'r') as outfile:
                self.config = json.load(outfile)
            self.config["p"]=[float(p) for p in self.config["p"]]
            self.reinit(self.config, formatter, selector)
            print(">>    Config load at {}".format(path_config))

        m = list(self.index_to_nodes.values())
        for i, elt in enumerate(m):
            elt.load_model(self.selector, file = file, name = 'classState'+str(i)+'_'+name)

        self.selector.load_model(file = file, name = 'selector_'+name)
        #self.formatter.load_model(file = file, name = 'formatter_'+name)

        print(">>    Model load in {}".format(file))

    # def render(self, waiting_time = 0.1, action = None, image_current= np.ones((200, 300)), image_class= np.ones((200, 300)), image_next= np.ones((200, 300))):
    #     self.viewer.plot(waiting_time = waiting_time, action = action, image_current= image_current, image_class=image_class, image_next= image_next)

    # def renderComplete(self, waiting_time = 0.1, action = [], current = np.ones((200, 300))):
    #
    #     module = self.modules_list[self.index_to_modules[self.old_node]]
    #     im_class = module.getOneElementBufferIm()
    #
    #     index_next = module.findBetterN()
    #     module_next = self.modules_list[self.index_to_modules[index_next]]
    #     im_next = module_next.getOneElementBufferIm()
    #
    #     self.viewer.G.target_node(self.old_node, index_next, reverse = False)
    #     self.render(waiting_time=waiting_time, action = action, image_current=current, image_class = im_class, image_next = im_next)
    #     self.viewer.G.target_node(self.old_node, index_next, reverse = True)


if __name__ == '__main__':
    config = {"d_input": 4, "d_hidden": 32, "d_output": 16,
            "action_dim": 8, "action_lim": [0, 1], "cont": True,
            "nodes": [[0, 1, 2], [0, 1, 2], [0, 1, 2]], "init_active": 2,
            "p": [0, 0, 1]}

    # from dataCartPol import load_images
    # img_gauche, img_droite, img_centre = load_images(path = "./Data")
    #
    lms = LMS(config)
    # for n in lms.modules_list:
    #     if n.index == 0:
    #         n.BufferIm+= img_gauche
    #     if n.index == 1:
    #         n.BufferIm+= img_droite
    #     if n.index == 2:
    #         n.BufferIm+= img_centre
    # waiting_time = 5
    #
    # lms.renderComplete(waiting_time = waiting_time)
    # for i in range(10):
    #     x = torch.randn(1, config["d_input"])
    #     id = lms.findActive(x)
    #     lms.updateActive(id)
    #     lms.renderComplete(waiting_time = waiting_time)

    print(lms)

    lms.deleteNode(0)
    lms.old_node = 1
    #lms.renderComplete(waiting_time = waiting_time)

    #print(lms)

    lms.save_model()
    del lms

    lms = lms()
    lms.load_model()
    #print(lms)
