# -*- coding: utf-8 -*-
"""Create the rnn model.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

import torch
import torch.nn as nn
from toolBox import BlockLSTM
from operator import itemgetter
from link import Link
import torch.nn.functional as F
from sac import SAC_Agent, MLPNetwork, InsideSac, EvaluatorSac, Transition
import random
import os
import json
import numpy as np
import networkx as nx



class ClassState(nn.Module):
    def __init__(self, d_state, d_output, d_hidden, action_dim, viewer, n_state, action_lim = None, use_evaluator = True, use_external_agent = True, use_inside_agent = True, use_LSTM_encoder = False, batchsize = [256, 256, 256], gamma = [0.99, 0.99, 0.99], reward_external= [-1, 2, -2], pre_learning = False, name = "ClassState0", view = "Pagg", learning_rate=1e-4, continuous = False, type_attention_evaluator = "H"):
        super(ClassState, self).__init__()

        self.action_lim = action_lim
        self.continuous = continuous
        self.use_LSTM_encoder = use_LSTM_encoder

        if view not in ['Pagg', 'Psucc']:
            view = 'Pagg'

        self.info = {"name":name, "view": view}

        self.d_hidden = d_hidden
        self.d_input = d_state

        if self.use_LSTM_encoder:
            self.d_output = d_output
        else:
            self.d_output = d_hidden

        self.action_dim = action_dim
        self.global_n_nodes = n_state

        self.attention_encoder = nn.Linear(d_hidden,d_output)
        self.LSTM = BlockLSTM(d_state, d_hidden)
        self.ht, self.ct = torch.randn(1, d_hidden), torch.randn(1, d_hidden)

        if self.use_LSTM_encoder:
            self.qt = torch.randn(1, d_output)
        else:
            self.qt = self.ht

        self.Psucc = 0
        self.Pagg = 0
        self.BufferIm = []
        self.neighbours = dict()
        self.edges = dict()

        self.viewer = viewer
        self.index = self.viewer.updateNodeG(0)
        self.updateOptimizer(learning_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.batchsize = batchsize
        self.gamma = gamma
        self.reward_external = reward_external
        self.pre_learning = pre_learning
        self.type_attention_evaluator = type_attention_evaluator

        self.use_evaluator = use_evaluator
        self.use_external_agent = use_external_agent
        self.use_inside_agent = use_inside_agent

    def updateOptimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam([
            {'params': self.attention_encoder.parameters()},
            {'params': self.LSTM.parameters()}
            ], lr=learning_rate)

    def create_encoded_neighbour(self, grl, cutoff=1):
        res = list(nx.single_source_shortest_path_length(grl.viewer.G.G, self.index, cutoff=cutoff).keys())
        nodes_neig = [grl.modules_list[i] for i in res]
        self.n_state = len(res)
        self.global_to_local = dict()
        self.H, self.Encoded_Id = [], []
        for i, n in enumerate(nodes_neig):
            encoded_index = F.one_hot(torch.tensor(i), self.n_state).float().view(1,-1)
            self.global_to_local.update({n.index: i})
            self.H.append(nodes_neig[i].compute_qt().detach())
            self.Encoded_Id.append(encoded_index)

        self.H = torch.cat(self.H, dim = 0)
        self.Encoded_Id = torch.cat(self.Encoded_Id, dim = 0)

    def update_agents_learning_rate(self, new_learning_rate, num = 0):
        agents = [self.evaluator, self.external_agent, self.inside_agent]
        agent = agents[num]
        agent.temp_optimizer.param_groups[0]['lr'] = new_learning_rate
        agent.q_optimizer.param_groups[0]['lr'] = new_learning_rate
        agent.policy_optimizer.param_groups[0]['lr'] = new_learning_rate


    def create_agents(self, grl, use_alpha = None):
         self.create_encoded_neighbour(grl)
         if self.use_evaluator:
             self.evaluator = EvaluatorSac(self.d_input, self.d_output, self.n_state, lr = 1e-4, batchsize = self.batchsize[0], hidden_size_policy=512, hidden_size_qfun=512, use_alpha = use_alpha, gamma = self.gamma[0], type_attention_evaluator = self.type_attention_evaluator)
             self.evaluator.createH(self.H, self.Encoded_Id)
         else:
             self.evaluator = None
             print(">>    No evaluator.")

         if self.use_external_agent:
             self.external_agent = SAC_Agent(self.d_input + self.n_state, self.action_dim, lr=1e-4, batchsize=self.batchsize[1], hidden_size=512, continuous = self.continuous, use_alpha=None, gamma=self.gamma[1], tau=5e-3)
         else:
             self.external_agent = None
             print(">>    No external agent.")

         if self.use_inside_agent:
             self.inside_agent = InsideSac(self.d_input, self.action_dim, lr=1e-4, batchsize=self.batchsize[2], hidden_size=512, continuous = self.continuous, use_alpha=None, gamma=self.gamma[2], tau=5e-3)
         else:
             self.inside_agent = None
             print(">>    No inside agent.")

    def print_info(self):
        print(">> INFORMATION ABOUT NODE ", self.index)
        print(">>     Evaluator: buffer size = ", self.evaluator.replay_pool.__len__())
        print(">>     External: buffer size = ", self.external_agent.replay_pool.__len__())
        print(">>     Inside: buffer size = ", self.inside_agent.replay_pool.__len__())
        return [self.evaluator.replay_pool.__len__(),  self.external_agent.replay_pool.__len__(), self.inside_agent.replay_pool.__len__()]


    def local_to_global_p(self, p):
        global_p = np.zeros(self.global_n_nodes)
        id = list(self.global_to_local.keys())
        global_p[id] = p
        return global_p

    def optimize_evaluator(self, n_eval):
        self.evaluator.optimize(n_eval)

    def get_eval_action(self, state, deterministic=True, random = False):
        self.evaluator.createH(self.H, self.Encoded_Id)
        if random:
            local_p = self.softmax(torch.randn((1, self.n_state)))
            local_p = local_p.numpy().tolist()[0]
        else:
            local_p = self.evaluator.get_action(state, deterministic=deterministic)

        global_p = self.local_to_global_p(local_p).tolist()
        return global_p, local_p

    def add_evaluator_transition(self, state, action, reward, next_state, done, node, id_goal, id_current, id_next, penality = -1):
        if (id_goal!= id_next and id_goal == id_current) and not (penality is None):
            r = penality
        else:
            r = reward
        node.evaluator.addTransition(state, action, r, next_state, done, self.index, id_next)


    def add_external_transition(self, batch_state, batch_action, batch_next_state, id_next):
        possible_goal_id = set(self.neighbours.keys())-set([self.index])
        if id_next in set(self.neighbours.keys()): #selector failure
            for possible_goal in possible_goal_id:
                goal =  self.Encoded_Id[self.global_to_local[possible_goal]].view(1, -1)
                for i in range(len(batch_state)-1):
                    s, s_next, action = batch_state[i], batch_next_state[i], batch_action[i]
                    state, next_state = torch.cat([s, goal], dim = -1).detach(), torch.cat([s_next, goal], dim = -1).detach()
                    reward, done = self.reward_external[0], False
                    if not self.continuous:
                        action = int(action)
                    self.external_agent.replay_pool.push(Transition(state, action, reward, next_state, done))

                s, s_next, action = batch_state[-1], batch_next_state[-1], batch_action[-1]
                state, next_state = torch.cat([s, goal], dim = -1).detach(), torch.cat([s_next, goal], dim = -1).detach()

                if possible_goal == id_next:
                    done, reward = True, self.reward_external[1]
                elif id_next == self.index:
                    done, reward = False, self.reward_external[0]
                else:
                    done, reward = True, self.reward_external[2]

                if not self.continuous:
                    action = int(action)
                self.external_agent.replay_pool.push(Transition(state, action, reward, next_state, done))

    def add_inside_transition(self, state, action, reward, next_state, done, id_next):
        if not self.continuous:
            action = int(action)
        self.inside_agent.addTransition(state, action, reward, next_state, done, self.index, id_next)

    def get_action(self, x, id_goal, deterministic = True, internal_strategy = None, external_strategy = None):
        if len(x.size())==1:
            x = x.view(1, -1)

        if id_goal == self.index and self.use_inside_agent:
            action = self.inside_agent.get_action(x, deterministic = deterministic)
        elif id_goal == self.index and not self.use_inside_agent:
            action = internal_strategy.get_action()
        elif id_goal != self.index and not self.use_external_agent:
            action = external_strategy.get_action()
        else:
            goal = self.Encoded_Id[self.global_to_local[id_goal]].view(1, -1)
            state = torch.cat([x, goal], dim=-1)
            action = self.external_agent.get_action(state, deterministic = deterministic)

        if self.action_lim is not None and self.continuous:
            action = (2/(self.action_lim[1]-self.action_lim[0]))*action + (1 - (2*self.action_lim[1])/(self.action_lim[1]-self.action_lim[0]))

        if not self.continuous:
            action = int(action)

        return action

    def notifyAgent(self, agent, remove = False):
        agent.notify(self.index, self.compute_qt(), remove)

    def getIndex(self):
        return self.index

    def addNeighbours(self, neighbours = []):
        for neighbour in neighbours:
            id = neighbour.getIndex()
            link = Link(self.index, id)
            self.viewer.updateEdgeG([(self.index, id)])

            self.neighbours.update({id: neighbour})
            self.edges.update({id: link})

    def delNeighbour(self, idx):
        self.neighbours.pop(idx)
        self.edges.pop(idx)

    def setPsucc(self, new_value):
         self.Psucc = new_value
         if self.info["view"] == "Psucc":
             self.viewer.updateProbaG([self.index], [new_value])

    def getPsucc(self):
         return self.Psucc

    def neighboursP(self, type = "Psucc"):
        idx = []
        p = []
        for n in self.neighbours.values():
            idx.append(n.getIndex())
            if type == "Psucc":
                p.append(n.getPsucc())
            else:
                p.append(n.getPagg())
        return idx, p

    def computePagg(self, k, reduction = 0.2):
        if k == 0:
            _, p = self.neighboursP(type="Psucc")
            if p!= []:
                new_value = max(reduction*max(p), self.getPsucc())
            else:
                new_value = self.getPsucc()
        else:
            _, p = self.neighboursP(type="Pagg")
            if p!= []:
                new_value = max((reduction*max(p), self.getPagg()))
            else:
                new_value = self.getPsucc()

        return new_value

    def setPagg(self, new_value):
        self.Pagg = new_value
        if self.info["view"] == "Pagg":
            self.viewer.updateProbaG([self.index], [self.Pagg])

    def getPagg(self):
        return self.Pagg

    def forward(self, x, h, c):
        if len(x.size())==1:
            x = x.view(1, -1)

        ht, ct = self.LSTM(x, h, c)

        qt = self.compute_qt(ht)

        return ht, ct, qt

    def compute_qt(self, ht = None, use_LSTM_encoder = False):
        if ht is None:
            ht = self.ht.detach()

        if self.use_LSTM_encoder or use_LSTM_encoder:
            qt = self.attention_encoder(ht)
        else:
            qt = ht

        return qt

    def updateMemory(self, ht, ct, qt):
        self.ht, self.ct, self.qt = ht, ct, qt

    def addBufferIm(self, x, max_len = 100):
        self.BufferIm.append(x)
        self.BufferIm = self.BufferIm[:max_len]

    def getOneElementBufferIm(self, idx = None):
        if self.BufferIm!= []:
            if idx is None:
                return random.choice(self.BufferIm)
            else:
                l = len(self.BufferIm)
                if l>=idx:
                    print(">>  ERROR: idx > #BufferIm, took idx%#BufferIm")
                return self.BufferIm[idx%l]
        else:
            return None

    def delete(self, agent):
        pred = list(self.viewer.G.G.predecessors(self.index))
        self.viewer.updateNodeG(0, remove = (True, self.index))
        self.notifyAgent(agent, remove = True)
        return pred

    def findBetterN(self, type = "Psucc"):
        idx, p = self.neighboursP(type = type)
        tuple = list(zip(idx, p))
        return max(tuple, key=itemgetter(1))[0]

    def save_model(self, file = './Results', name = 'classState_latest'):
        path_model = file + '/'
        path_data = file + '/' + name +'.txt'
        os.makedirs(file , exist_ok=True)

        torch.save(self.attention_encoder.state_dict(), path_model + name + "_attention_encoder"+".pth")
        torch.save(self.LSTM.state_dict(), path_model + name + "_LSTM"+".pth")

        if self.use_external_agent:
            self.external_agent.save_model(file = file, name = name+"_external_agent")
        if self.use_inside_agent:
            self.inside_agent.save_model(file = file, name = name+"_inside_agent")
        if self.use_evaluator:
            self.evaluator.save_model(file = file, name = name+"_evaluator")

        internal_data = {"qt": self.qt.detach().tolist(),
                         "ht": self.ht.detach().tolist(),
                         "ct": self.ct.detach().tolist()}
        with open(path_data, 'w') as outfile:
            json.dump(internal_data, outfile)

        print(">>    Model saved at {}".format(path_model+ name))
        print(">>    Internal Data saved at {}".format(path_data))

    def load_model(self,agent, file = './Results', name = 'classState_latest'):
        path_model = file + '/'
        path_data = file + '/' + name +'.txt'

        self.attention_encoder.load_state_dict(torch.load(path_model + name +"_attention_encoder"+ ".pth"), strict=False)
        self.LSTM.load_state_dict(torch.load(path_model + name + "_LSTM"+".pth"), strict=False)

        if self.use_external_agent:
            self.external_agent.load_model(file = file, name = name+"_external_agent")
        if self.use_inside_agent:
            self.inside_agent.load_model(file = file, name = name+"_inside_agent")
        if self.use_evaluator:
            self.evaluator.load_model(file = file, name = name+"_evaluator")

        with open(path_data, 'r') as outfile:
            internal_data = json.load(outfile)

        h = torch.tensor(internal_data["ht"])
        c = torch.tensor(internal_data["ct"])
        q = torch.tensor(internal_data["qt"])
        self.updateMemory(h, c, q)
        self.notifyAgent(agent)

        print(">>    Model loaded at {}".format(path_model))
        print(">>    Internal Data loaded at {}".format(path_data))


if __name__ == '__main__':
    from visualization import Viewer

    viewer = Viewer()
    d_v, d_q, d_hidden = 4, 6, 8
    CS0 = ClassState(d_v, d_q, d_hidden, viewer, name = "ClassState0")
    CS1 = ClassState(d_v, d_q, d_hidden, viewer, name = "ClassState1")
    viewer.plot(waiting_time = 2)
    print(">>   Get index:"+CS0.info['name']+" = " + str(CS0.getIndex()) + " - " + CS1.info['name'] + " = " + str(CS1.getIndex()))

    CS0.addNeighbours([CS1])
    viewer.plot(waiting_time = 2)
    print(">>  Neighbours of CS0:", CS0.neighbours, " and of CS1:", CS1.neighbours)

    CS1.setPsucc(1)
    print(">>  Psucc for CS0:", CS0.getPsucc(), "and for CS1:", CS1.getPsucc())
    viewer.plot(waiting_time = 2)

    n_CS0 = CS0.computePagg(k=0)
    n_CS1 = CS1.computePagg(k=0)

    CS0.setPagg(n_CS0)
    CS1.setPagg(n_CS1)

    print(">>  Pagg for CS0:", CS0.getPagg(), "and for CS1:", CS1.getPagg())
    viewer.plot(waiting_time = 2)

    n_CS0 = CS0.computePagg(k=1)
    n_CS1 = CS1.computePagg(k=1)

    CS0.setPagg(n_CS0)
    CS1.setPagg(n_CS1)
    print(">>  Pagg for CS0:", CS0.getPagg(), "and for CS1:", CS1.getPagg())
    viewer.plot(waiting_time = 2)


    x = torch.randn(1, d_hidden)
    v = torch.randn(1, d_v)
    new_h = CS1(v)
    CS1.addBufferIm(x)
    print(">>  Notification at:", at.nodeQ)
    print(">>  new_h:", new_h)
    print(">>  BufferIm:", CS1.BufferIm, x)

    class Agent():
        def __init__(self):
            pass
        def notify(self, index, qt, remove = False):
            print(">> Notify:", index, " - ", qt, " - ", remove)

    ag = Agent()

    succ = CS1.delete(ag)
    del CS1
    viewer.plot(waiting_time = 2)
    print(">>  Notification at:", at.nodeQ)
