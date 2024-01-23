# -*- coding: utf-8 -*-
"""Test the learning with CartStemContact example.
States and edges are hand-defined.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

n_state = 3
PATH_DATA = "./Data/CartStemContact"
PATH_SAVE = './Results/CartStemContact/meta_states'

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from lms import LMS
from dataCartStemContact import CartStemContactDatasetState, get_data_loader
import random
import numpy as np
import json


from agent import Formatter, Evaluator, Selector

print("\n #### START ####\n")

seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
K = 0

def evaluate_agent(env, lms, n_starts=1, render = False):
    reward_sum = 0
    liste_reward = []

    for n in lms.modules_list:
        n.external_agent.eval()
        n.inside_agent.eval()
        n.evaluator.eval()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = False

    for i in range(n_starts):
        done = False
        state = env.reset()
        observation = torch.tensor(state).view(1,-1).type(torch.float)
        r = 0

        inOption = False
        while (not done):
            id, node = lms.findActive(observation)
            state = observation.view(1,-1)

            if not inOption:
                action_evaluator, _ = node.get_eval_action(state, random = False, deterministic=True)
                lms.updateP(action_evaluator, K)
                goal_id = node.findBetterN()

            if goal_id != id:
                inOption = True

            action = node.get_action(observation, goal_id, deterministic=True)

            nextstate, reward, done, _ = env.step(action)

            if render:
                env.render()
            reward_sum += reward
            r += reward
            state = nextstate
            observation = torch.tensor(state).view(1,-1).type(torch.float)

            next_id, _ = lms.findActive(observation)
            if next_id != id:
                inOption = False
        print(">> Step: ", i, " - current reward:", r, " - mean reward:", reward_sum/(i+1))
        liste_reward.append(r)

    for n in lms.modules_list:
        n.external_agent.train()
        n.inside_agent.train()
        n.evaluator.train()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = True

    return reward_sum / n_starts, liste_reward
def evaluate_agent_QP(env, envQP, lms, n_starts=1, render = False):
    reward_sum = 0
    liste_reward = []

    for n in lms.modules_list:
        n.external_agent.eval()
        n.evaluator.eval()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = False

    for i in range(n_starts):
        done = False
        state = env.reset()
        observation = torch.tensor(state).view(1,-1).type(torch.float)
        r = 0

        inOption, inQP, done_QP = False, False, False
        while (not done):
            id, node = lms.findActive(observation)
            state = observation.view(1,-1)

            if not inOption: # and not inQP:
                action_evaluator, local_action_evaluator = node.get_eval_action(state, random = False, deterministic=True)
                lms.updateP(action_evaluator, K)
                goal_id = node.findBetterN()


            if goal_id!=id and not inOption and not inQP:
                inOption = True
                inQP = False
            elif goal_id==id and not inQP and not inOption:
                inQP = True
                pos = env.get_position()
                config = {"goalList": env.config["goalList"], "init_x": env.config["init_x"],
                                "cube_x": env.config["cube_x"], "max_move": env.config["max_move"],
                                "goalPos": env.config["goalPos"], "pos": pos}
                QPenv.reset(config)


            if not inQP:
                action = node.get_action(observation, goal_id, deterministic = True)
            else:
                action, objective_QP, done_QP, _ = QPenv.step([])


            nextstate, reward, done, _ = env.step(action)

            if render:
                env.render()

            reward_sum += reward
            r += reward
            state = nextstate
            observation = torch.tensor(state).view(1,-1).type(torch.float)

            next_id, _ = lms.findActive(observation)


            if next_id!= id and inOption:
                inOption = False
            if inQP and (next_id!= id or done_QP):
                inQP = False

        print(">> Step: ", i, " - current reward:", r, " - mean reward:", reward_sum/(i+1))
        liste_reward.append(r)

    for n in lms.modules_list:
        n.external_agent.train()
        n.evaluator.train()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = True

    return reward_sum / n_starts, liste_reward
def evaluate_agent_mixte(env, envQP, lms, liste_learned = [], n_starts=1, render = False):
    reward_sum = 0
    liste_reward = []

    for n in lms.modules_list:
        n.external_agent.eval()
        n.evaluator.eval()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = False

    for i in range(n_starts):
        done = False
        state = env.reset()
        observation = torch.tensor(state).view(1,-1).type(torch.float)
        r = 0

        inOption, inQP, done_QP = False, False, False
        while (not done):
            id, node = lms.findActive(observation)
            state = observation.view(1,-1)

            if not inOption:
                action_evaluator, local_action_evaluator = node.get_eval_action(state, random = False, deterministic=True)
                lms.updateP(action_evaluator, K)
                goal_id = node.findBetterN()


            if goal_id!=id and not inOption and not inQP:
                inOption = True
                inQP = False
            elif goal_id==id and not inQP and not inOption and id not in liste_learned:
                inQP = True
                pos = env.get_position()
                config = {"goalList": env.config["goalList"], "init_x": env.config["init_x"],
                                "cube_x": env.config["cube_x"], "max_move": env.config["max_move"],
                                "goalPos": env.config["goalPos"], "pos": pos}
                QPenv.reset(config)


            if not inQP or id in liste_learned:
                action = node.get_action(observation, goal_id, deterministic = True)
            else:
                action, objective_QP, done_QP, _ = QPenv.step([])


            nextstate, reward, done, _ = env.step(action)

            if render:
                env.render()

            reward_sum += reward
            r += reward
            state = nextstate
            observation = torch.tensor(state).view(1,-1).type(torch.float)

            next_id, _ = lms.findActive(observation)


            if next_id!= id and inOption:
                inOption = False
            if inQP and (next_id!= id or done_QP):
                inQP = False

        print(">> Step: ", i, " - current reward:", r, " - mean reward:", reward_sum/(i+1))
        liste_reward.append(r)

    for n in lms.modules_list:
        n.external_agent.train()
        n.evaluator.train()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = True

    return reward_sum / n_starts, liste_reward

class MyFormatter(Formatter):
    def __init__(self):
        super(MyFormatter, self).__init__()
    def createRepresentation(self, x, g):
        return x

nodes = [[0, 1], [0, 1, 2], [1, 2]]
print("CONNECTIVITY:", nodes)
p = [0 for _ in range(n_state)]

config = {"d_input": 8, "d_hidden": 64, "d_output": 64,
        "action_dim": 1, "action_lim": [-1, 1], "continous": True,
        "nodes": nodes, "init_active": int(n_state-1)/2,
        "p": p, "batchsize": [128, 128, 128], "gamma": [0.99, 0.99, 0.99],
        "reward_external": [-1, 20, -2], "use_LSTM_encoder": False, "pre_learning": False,
        "use_agents": [True, True, True], "type_attention_evaluator": "H"}

m_formatter = MyFormatter()
m_selector = Selector(config["d_input"], config["d_output"], learning_rate=1e-4)
lms = LMS(m_formatter, m_selector, config=config)
lms.load_model(m_formatter, m_selector, file = PATH_SAVE)

from dataCartStemContact import CartStemContactDatasetState, get_data_loader
print("VERIFICATION LOAD")
dataset_validation = CartStemContactDatasetState(path = PATH_DATA, validation = True, n_state = n_state)
print(">> Validation dataset state len:", dataset_validation.__len__())
dataloader_validation = get_data_loader(dataset_validation, dataset_validation.__len__(), True)
l_data_validation = len(dataloader_validation)
for i, data in enumerate(dataloader_validation):
    state, id_ = data
    correct, non_correct = lms.evalHandwritten(state, id_)
    tot = correct+ non_correct
    print(i,"/", l_data_validation, "   >>>   Validation:", 100*correct/tot, "% correct - ", 100*non_correct/tot, "% non-correct")

print(">> Architecture")
for n in lms.modules_list:
    print("Index:", n.index, " - Neighbours:", n.neighbours.keys(), " - ht:", n.ht)
for p in lms.selector.parameters():
    p.requires_grad = False
for n in lms.modules_list:
    for p in n.LSTM.parameters():
        p.requires_grad = False
    for p in n.attention_encoder.parameters():
        p.requires_grad = False

print(">> CONFIGURATION")
for k in lms.config.keys():
    print(">>    ", k, ":", lms.config[k])



import gym
import sofagym

env = gym.make("cartstemcontact-v2")
QPenv = gym.make('qpcartstemcontact-v0')

env.configure({"render":1})
env.configure({"visuQP":False})


# print("\n\n####################   START LMS  ######################")
# reward_lms, _ = evaluate_agent(env, lms, n_starts=1000, render = False)
# print("\n\n####################   START QP  ######################")
# reward_QP, _ = evaluate_agent_QP(env, QPenv, lms, n_starts=1000, render = False)
# print("\n\n####################   RESULTS  ######################")
# print(">>  Reward LMS:", reward_lms)
# print(">>  Reward QP:", reward_QP)
#

print("\n\n####################   START MIXTE 1 not learned  ######################")
reward_0_2_learned, _ = evaluate_agent_mixte(env, QPenv, lms, liste_learned = [0, 2], n_starts=1000, render = False)
print("\n\n####################   START MIXTE 0/2 not learned  ######################")
reward_1_learned, _ = evaluate_agent_mixte(env, QPenv, lms, liste_learned = [1], n_starts=1000, render = False)
print("\n\n####################   RESULTS  ######################")
print(">>  Reward 1 not learned :", reward_0_2_learned)
print(">>  Reward 0/2 not learned:", reward_1_learned)
