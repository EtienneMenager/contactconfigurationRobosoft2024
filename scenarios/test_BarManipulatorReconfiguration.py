# -*- coding: utf-8 -*-
"""Test the learning with CartStemContact example.
States and edges are hand-defined.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

PENALITY = -5
K = 0

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-lm", "--list_ms", help="The num of the test we want to plot",
                     action='append', required = True)
parser.add_argument("-b", "--batchsize", help = "batchsize for the agents",
                     action='append', type = int)
parser.add_argument("-g", "--gamma", help = "gamma for the agents",
                    action='append', type=float)
parser.add_argument("-ts", "--train_states", help = "Train the selector",
                    action="store_true")
parser.add_argument("-ta", "--train_actions", help = "Train the agents",
                    action="store_true")
parser.add_argument("-ne", "--n_epoch", help = "Number of epochs for the training of actions",
                    type=int, default = 250000)
parser.add_argument("-ft", "--final_trajectory", help = "Use the final trajectory for the external agent",
                    action="store_true")
parser.add_argument("-nr", "--n_random", help = "Number of random actions before to train evaluator",
                    type=int, default = 500)
parser.add_argument("-re", "--reward_external", help = "Reward for the external agent (3 elements: reward in the meta-states, good reward, bad reward)",
                     action='append', type = int)
parser.add_argument("-nt", "--n_train", help="Numer of train (3 elements: external, inside, evaluator)",
                     action='append', type = int)
parser.add_argument("-d", "--determinist", help = "Use determinist agent to train evaluators",
                    action="store_true")
parser.add_argument("-bc", "--behavior_cloning", help = "Force the evaluator to follow a given trajectory",
                    action="store_true")
args = parser.parse_args()

TRAIN_STATES = args.train_states
TRAIN_ACTION = args.train_actions
N_EPOCH = args.n_epoch
RANDOM = args.n_random

if args.n_train is None:
    N_TRAIN = [5, 5, 5]
elif len(args.n_train)!=3:
    N_TRAIN = [5, 5, 5]
else:
    N_TRAIN = args.n_train

n_state = 5


import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from lms import LMS
import random
import numpy as np
import json
import os


from agent import Formatter, Evaluator, Selector

num = args.list_ms[0]
seed = int(num)*10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

PATH_DATA = "./Data/BarManipulator"
PATH_SAVE = './Results/BarManipulator/meta_states/seed_' + str(seed)
os.makedirs(PATH_SAVE, exist_ok=True)

print("\n #### START " + num + "####\n")

def evaluate_agent(env, lms, n_starts=1, render = False, goal = None):
    reward_sum = 0
    reconfiguration = Reconfiguration()
    liste_reward = []
    final_reward = []

    for n in lms.modules_list:
        n.inside_agent.eval()
        n.evaluator.eval()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = False

    for _ in range(n_starts):
        done = False
        state = env.reset(goal = goal)
        reconfiguration.reset()
        observation = torch.tensor(state).view(1,-1).type(torch.float)
        r = 0

        inOption = False
        step = 1
        while (not done):
            get_ms = env.get_infos()[0]
            id, node = lms.modules_list[get_ms].index, lms.modules_list[get_ms]
            state = observation.view(1,-1)

            if not inOption:
                action_evaluator, _ = node.get_eval_action(state, random = False, deterministic=True)
                lms.updateP(action_evaluator, K)
                goal_id = node.findBetterN()

            if goal_id != id:
                inOption = True
                # reconfiguration.update_dir(observation[0][-2], observation[0][-1], goal_id)
                reconfiguration.update_dir(observation[0][-2], observation[0][-1], get_ms)

            action = node.get_action(observation, goal_id, deterministic=True, external_strategy = reconfiguration)

            nextstate, reward, done, _ = env.step(action)

            if render:
                env.render()
                print(">> Epoch: ", step, " - Reward = ", reward)
            step+=1
            reward_sum += reward
            r += reward
            state = nextstate
            observation = torch.tensor(state).view(1,-1).type(torch.float)

            if reconfiguration.get_id() == 0:
                inOption = False


        liste_reward.append(r)
        final_reward.append(reward)

    for n in lms.modules_list:
        n.inside_agent.train()
        n.evaluator.train()
    for n in lms.modules_list:
        for p in n.parameters():
            p.requires_grad = True

    return reward_sum / n_starts, liste_reward, final_reward

class MyFormatter(Formatter):
    def __init__(self):
        super(MyFormatter, self).__init__()
    def createRepresentation(self, x, g):
        return x

### TOT CONNECTION 1=8 ; 2=10; 3=9; 4 = 11
"""
Configuration:

                2 |       |1
                  |       |
                  |       |
                  
                  --------- bar

0: no contact         1: cross, 1 top         2: no cross, 1 top        3: cross, 2 top           4: no cros, 2 top

          1
                         1                                  1                       2               2
-----------           ----------------          ---------------         --------------              ----------------
                                    2               2                       1                                   1
                                    
2

Action:

action = [a, b, c, d, e, f, g, h, i]
a: finger 1, end part, +1 beside -1 front
b: finger 1, end part, +1 left -1 right
c: finger 1, top part, +1 beside -1 front
d: finger 1, top part, +1 left -1 right
e: finger 2, end part, +1 beside -1 front
f: finger 2, end part, +1 left -1 right
g: finger 2, top part, +1 beside -1 front
h: finger 2, top part, +1 left -1 right
i: base, +1 up -1 down 

strat = [[0, 0, 0.7, -0.7, 0, 0, -0.7, 0.7, -1],

[0, 0, -0.7, -0.5, 0, 0, 0.7, 0.5, -1],

[0, 0, -0.3, -0.5, 0, 0, 0.3, 0.5, 0.8],

[0, 0, -0.3, -0.3, 0, 0, 0.3, 0.3, 0.8],
]




"""

# nodes = [[0, 2, 4],
#          [1, 4],
#          [2],
#          [3, 2],
#          [4]]

# nodes = [[0, 4],
#          [1, 4],
#          [2, 4],
#          [3, 2],
#          [4, 2]]

nodes = [[0, 4],
         [1, 4],
         [2],
         [3, 2],
         [4]]



class Reconfiguration:
    def __init__(self):
        self.id = 0
        self.dir = None
        self.ms = None

    # def update_dir(self, curr_pos, goal, goal_id):
    #     if curr_pos > goal or goal_id == 4:
    #         self.dir = -1
    #     else:
    #         self.dir = 1

    def update_dir(self, curr_pos, goal, ms):
        if curr_pos > goal:
            self.dir = -1
        else:
            self.dir = 1
        self.ms = ms

    def get_id(self):
        return self.id

    def reset(self):
        self.id = 0
        self.dir = None
        self.ms = None

    def get_action(self):
        if self.ms == 3:
            if self.dir == 1:
                action_1 = [0, 0, -0.7, -0.7, 0, 0, 0.7, 0.7, -1]
                action_2 = [0, 0, +0.7, 1, 0, 0, -0.7, -1, -0.7]
                action_3 = [0, 0, +0.3, 1, 0, 0, -0.3, -1, 0.8]
                action_4 = [0, 0, 0.3, 0.3, 0, 0, -0.3, -0.3, 1]
            else:
                action_1 = [0, 0, -0.7, -0.7, 0, 0, 0.7, 0.7, -1]
                action_2 = [0, 0, +0.7, -0.5, 0, 0, -0.7, 0.5, -1]
                action_3 = [0, 0, +0.3, -0.3, 0, 0, -0.3, 0.3, 0.8]
                action_4 = [0, 0, 0.3, 0.1, 0, 0, -0.3, -0.1, 0.8]
        elif self.ms == 1:
            if self.dir == 1:
                action_1 = [0, 0, 0.7, -0.7, 0, 0, -0.7, 0.7, -1]
                action_2 = [0, 0, -0.7, -0.5, 0, 0, 0.7, 0.5, -1]
                action_3 = [0, 0, -0.3, -0.3, 0, 0, 0.3, 0.3, 0.8]
                action_4 = [0, 0, -0.3, 0.1, 0, 0, 0.3, -0.1, 0.8]
            else:
                action_1 = [0, 0, 0.7, -0.7, 0, 0, -0.7, 0.7, -1]
                action_2 = [0, 0, -0.7, 1, 0, 0, 0.7, -1, -0.7]
                action_3 = [0, 0, -0.3, 1, 0, 0, 0.3, -1, 0.8]
                action_4 = [0, 0, -0.3, 0.3, 0, 0, 0.3, -0.3, 1]
        else:
            if self.dir == 1:
                action_1 = [0, 0, -1, 0, 0, 0, +1, 0, -1]
                action_2 = [0, 0, +0.7, 0, 0, 0, -0.7, 0, -1]
                action_3 = [0, 0, +0.7, 0, 0, 0, -0.7, 0, 0.7]
                action_4 = [0, 0, +0.3, 0, 0, 0, -0.3, 0, 0.7]
            else:
                action_1 = [0, 0, +1, 0, 0, 0, -1, 0, -1]
                action_2 = [0, 0, -0.7, 0, 0, 0, +0.7, 0, -1]
                action_3 = [0, 0, -0.7, 0, 0, 0, +0.7, 0, 0.7]
                action_4 = [0, 0, -0.3, 0, 0, 0, +0.3, 0, 0.7]


        init_action = []
        init_action.append(action_1)
        init_action.append(action_1)
        init_action.append(action_2)
        init_action.append(action_2)
        init_action.append(action_3)
        init_action.append(action_3)
        init_action.append(action_4)
        init_action.append(action_4)

        action = init_action[self.id]
        self.id = (self.id +1)%len(init_action)
        return np.array(action)



print("CONNECTIVITY:", nodes)
p = [0 for _ in range(n_state)]

config = {"d_input": 23, "d_hidden": 10, "d_output": 10,
        "action_dim": 9, "action_lim": [-1, 1], "continous": True,
        "nodes": nodes, "init_active": 1,
        "p": p, "batchsize": args.batchsize, "gamma": args.gamma,
        "reward_external": args.reward_external, "use_LSTM_encoder": False, "pre_learning": False,
        "use_agents": [True, False, True], "type_attention_evaluator": "H"}
m_formatter = MyFormatter()
m_selector = Selector(config["d_input"], config["d_output"], learning_rate=1e-4, hidden_size = 512)

lms = LMS(m_formatter, m_selector, config=config)


if TRAIN_ACTION:
     with torch.autograd.set_detect_anomaly(True):
        import gym
        import sofagym
        #lms.load_model(m_formatter, m_selector, file = PATH_SAVE)

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

        env = gym.make("barmanipulator-v0")
        env_test = gym.make("barmanipulatortest-v0")
        env.configure({"render":0})
        reconfiguration = Reconfiguration()

        max_steps = env.config["timer_limit"]
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        observation = env.reset()
        reconfiguration.reset()
        observation = torch.tensor(observation).type(torch.float)

        r, best = 0, -10000
        n_epoch = N_EPOCH
        freq_val = 600
        final_reward_list, reward_list, step = [], [], []
        print("\n>> START LEARNING WITH RECONFIGURATION: \n")

        n_random_actions = RANDOM
        time_step=0
        actual_random = [0 for _ in range(len(nodes))]

        inOption, rewardOption, stateOption, len_option = False, np.array([]), [], 0
        deterministic = True

        for e in range(1, n_epoch):
            time_step+=1

            get_ms = env.get_infos()[0]
            id, node = lms.modules_list[get_ms].index, lms.modules_list[get_ms]
            state = observation.view(1,-1)
            old_observation = observation.view(1, -1)

            if not inOption:
                action_evaluator, local_action_evaluator = node.get_eval_action(state, random = e<n_random_actions, deterministic=False)
                lms.updateP(action_evaluator, K)
                goal_id = node.findBetterN()

            if goal_id!=id and not inOption:
                inOption = True
                optionId, optionNode = id, node
                stateOption = []
                rewardOption = np.array([])
                len_option = 0
                # reconfiguration.update_dir(observation[-2], observation[-1], goal_id)
                reconfiguration.update_dir(observation[-2], observation[-1], get_ms)


            observation = lms.formatter(observation, None)

            if actual_random[id] < n_random_actions:
                if (inOption and np.random.randint(7)==0) or reconfiguration.get_id() != 0:
                    action = node.get_action(observation, goal_id, deterministic=(deterministic and args.determinist),
                                             external_strategy=reconfiguration)
                else:
                    inOption = False
                    reconfiguration.reset()
                    action = env.action_space.sample().tolist()
                actual_random[id] += 1
            else:
                action = node.get_action(observation, goal_id, deterministic = (deterministic and args.determinist), external_strategy = reconfiguration)

            observation, reward, done, info = env.step(action)
            observation = torch.tensor(observation).type(torch.float)

            # env.render()
            r+= reward
            get_ms = env.get_infos()[0]
            next_id, next_node = lms.modules_list[get_ms].index, lms.modules_list[get_ms]
            nextstate = observation.view(1,-1)

            real_done = False if time_step == max_steps else done
            reward_evaluator = reward
            if inOption:
                rewardOption = np.append(rewardOption, [0])
                rewardOption+= reward_evaluator
                stateOption.append(state)
                len_option+=1

            if actual_random[id] > n_random_actions:
                if inOption and reconfiguration.get_id() == 0 and deterministic:
                    # for i in range(len_option):
                    #     sOpt, rOpt = stateOption[i], rewardOption[i]
                    #     optionNode.add_evaluator_transition(sOpt, local_action_evaluator, rOpt, nextstate, real_done, optionNode, goal_id, optionId, next_id, penality = None)
                    optionNode.add_evaluator_transition(stateOption[0], local_action_evaluator, rewardOption[0], nextstate, real_done, optionNode, goal_id, optionId, next_id, penality=None)

                elif not inOption and deterministic:
                    node.add_evaluator_transition(state, local_action_evaluator, reward_evaluator, nextstate, real_done, node, goal_id, id, next_id, penality = None)

            if inOption and reconfiguration.get_id() == 0:
                inOption = False

            node.add_inside_transition(state, action, reward, nextstate, real_done, next_id)


            if done:
                if inOption and e > n_random_actions and deterministic:
                #     for i in range(len_option):
                #         sOpt, rOpt = stateOption[i], rewardOption[i]
                #         optionNode.add_evaluator_transition(sOpt, local_action_evaluator, rOpt, nextstate, real_done, optionNode, goal_id, optionId, next_id, penality = None)
                    optionNode.add_evaluator_transition(stateOption[0], local_action_evaluator, rewardOption[0], nextstate, real_done, optionNode, goal_id, optionId, next_id, penality=None)

                observation = env.reset()
                reconfiguration.reset()
                observation = torch.tensor(observation).type(torch.float)
                print(seed, ":", e, "/", n_epoch, "  >REWARD:", r)
                print(">> number random:", actual_random, "over ", n_random_actions)
                r=0
                time_step = 0
                inOption, rewardOption, stateOption, len_option = False, np.array([]), [], 0
                lms.optimize_agents_inside(N_TRAIN[1])
                lms.optimize_evaluator(N_TRAIN[2])

                if deterministic and args.determinist:
                    deterministic = False
                else:
                    deterministic = True

            if e%freq_val==0:
                current_reward, current_list_reward, current_final_reward = evaluate_agent(env_test, lms, n_starts=5, render = False)

                print(seed, ":", "VALIDATION:", current_reward)
                reward_list += current_list_reward
                final_reward_list += current_final_reward
                step += [e for _ in range(len(current_list_reward))]

                print(PATH_SAVE+"/rewards_"+num+".txt")
                with open(PATH_SAVE+"/rewards_"+num+".txt", 'w') as fp:
                   json.dump([reward_list, step, [], final_reward_list], fp)

                if current_reward >= best:
                    best = current_reward
                    print(">> BEST SCORE")
                    lms.save_model(file = PATH_SAVE)

        #lms.save_model(file = PATH_SAVE)

print("TEST")
lms.load_model(m_formatter, m_selector, file = PATH_SAVE)
import gym
import sofagym

env = gym.make("barmanipulator-v0")
env.config.update({"render":2})
env.config.update({"visuQP":True})
evaluate_agent(env, lms, n_starts=1, render = True, goal = 230)


