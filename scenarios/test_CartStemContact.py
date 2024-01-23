# -*- coding: utf-8 -*-
"""Test the learning with CartStemContact example.
States and edges are hand-defined.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

PENALITY = -1
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

num = args.list_ms[0]
seed = int(num)*10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

print("\n #### START " + num + "####\n")

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

    for _ in range(n_starts):
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

        liste_reward.append(r)

    for n in lms.modules_list:
        n.external_agent.train()
        n.inside_agent.train()
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
        "p": p, "batchsize": args.batchsize, "gamma": args.gamma,
        "reward_external": args.reward_external, "use_LSTM_encoder": False, "pre_learning": False,
        "use_agents": [True, True, True], "type_attention_evaluator": "H"}

m_formatter = MyFormatter()
m_selector = Selector(config["d_input"], config["d_output"], learning_rate=1e-4)

lms = LMS(m_formatter, m_selector, config=config)

if TRAIN_STATES:
    #Load data
    dataset_train = CartStemContactDatasetState(path = PATH_DATA, validation = False, n_state = n_state)
    dataset_validation = CartStemContactDatasetState(path = PATH_DATA, validation = True, n_state = n_state)

    print(">> Train dataset state len:", dataset_train.__len__())
    print(">> Validation dataset state len:", dataset_validation.__len__())

    n_epoch = 10000
    batch_size = 64
    freq_val = 250

    dataloader_train = get_data_loader(dataset_train, batch_size, True)
    dataloader_validation = get_data_loader(dataset_validation, dataset_validation.__len__(), True)

    l_data_train = len(dataloader_train)
    l_data_validation = len(dataloader_validation)


    with torch.autograd.set_detect_anomaly(True):
        # lms.load_model(m_formatter, m_selector, file = PATH_SAVE)
        #
        # print("VERIFICATION LOAD")
        # for i, data in enumerate(dataloader_validation):
        #     state, id_ = data
        #     correct, non_correct = lms.evalHandwritten(state, id_)
        #     tot = correct+ non_correct
        #     print(i,"/", l_data_validation, "   >>>   Validation:", 100*correct/tot, "% correct - ", 100*non_correct/tot, "% non-correct")
        # old_val = 100*correct/tot

        old_val = 0

        for e in range(n_epoch):
            loss = 0
            print(">>>   Start epoch ", e, "/", n_epoch)
            for i, data in enumerate(dataloader_train):
                state, id_ = data
                l = lms.one_step_learningHandWritten(state, id_)

                if i%5 == 0:
                    print(i,"/", l_data_train, "   >>>   Loss is:", l)

                if i%freq_val==0:
                    lms.eval()
                    for j, data in enumerate(dataloader_validation):
                        state, id_ = data
                        correct, non_correct = lms.evalHandwritten(state, id_)
                        tot = correct+ non_correct
                        p_correct = 100*correct/tot
                        p_non_correct = 100*non_correct/tot
                        print(j,"/", l_data_validation, "   >>>   Validation:", p_correct, "% correct - ", p_non_correct, "% non-correct")

                    lms.train()

                    if old_val < p_correct:
                        old_val = p_correct
                        lms.save_model( file = PATH_SAVE)

                        for n in lms.modules_list:
                            print("Index:", n.index," - ht:", n.ht)

if TRAIN_ACTION:
     with torch.autograd.set_detect_anomaly(True):
        lms.load_model(m_formatter, m_selector, file = PATH_SAVE)
        import gym
        import sofagym

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

        env = gym.make("cartstemcontact-v2")
        env_test = gym.make("cartstemcontacttest-v2")

        # env.configure({"render":1})
        # env.configure({"visuQP":True})

        max_steps = env.config["timer_limit"]
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        observation = env.reset()
        observation = torch.tensor(observation).type(torch.float)

        r, best = 0, -20
        n_epoch = N_EPOCH
        freq_val = 600
        reward_list, step = [], []
        print("START LEARNING")

        n_random_actions = RANDOM
        time_step=0

        inside, external, tot = [0], [0], [0]
        batch_state, batch_nextstate, batch_action = [], [], []

        inOption, rewardOption, stateOption, len_option = False, np.array([]), [], 0
        deterministic = True

        info_buffers = [-1, -1, -1]
        test_buffer_size = [False, False, False]
        buffer_size_goal = args.batchsize

        for e in range(1, n_epoch):
            time_step+=1
            id, node = lms.findActive(observation)

            state = observation.view(1,-1)
            old_observation = observation.view(1, -1)

            if not inOption:
                action_evaluator, local_action_evaluator = node.get_eval_action(state, random = e<n_random_actions, deterministic=False)
                lms.updateP(action_evaluator, K)
                goal_id = node.findBetterN()

            if goal_id!=id and not inOption:
                inOption = True
                stateOption = []
                rewardOption = np.array([])
                len_option = 0

            if e > n_random_actions:
                if id == goal_id:
                    inside.append(inside[-1]+1)
                    external.append(external[-1])
                    tot.append(tot[-1]+1)
                else:
                    inside.append(inside[-1])
                    external.append(external[-1]+1)
                    tot.append(tot[-1]+1)

            observation = lms.formatter(observation, None)

            if e < n_random_actions:
                action = env.action_space.sample().tolist()
            else:
                action = node.get_action(observation, goal_id, deterministic = (deterministic and args.determinist))
            observation, reward, done, info = env.step(action)
            observation = torch.tensor(observation).type(torch.float)
            real_done = False if time_step == max_steps else done
            # env.render()
            r+= reward
            next_id, next_node = lms.findActive(observation)
            nextstate = observation.view(1,-1)

            if inOption:
                rewardOption = np.append(rewardOption, [0])
                rewardOption+= reward
                stateOption.append(state)
                len_option+=1

            if e > n_random_actions:
                if next_id!= id and inOption and deterministic:
                    for i in range(len_option):
                        sOpt, rOpt = stateOption[i], rewardOption[i]
                        node.add_evaluator_transition(sOpt, local_action_evaluator, rOpt, nextstate, real_done, node, goal_id, id, next_id, penality = PENALITY)

                elif not inOption and deterministic:
                    node.add_evaluator_transition(state, local_action_evaluator, reward, nextstate, real_done, node, goal_id, id, next_id, penality = PENALITY)

            if next_id!= id and inOption:
                inOption = False

            node.add_inside_transition(state, action, reward, nextstate, real_done, next_id)

            batch_state.append(state)
            batch_nextstate.append(nextstate)
            batch_action.append(action)
            if next_id != id:
                node.add_external_transition(batch_state, batch_action, batch_nextstate, next_id)
                batch_state, batch_nextstate, batch_action = [], [], []

            if done:
                if inOption and e > n_random_actions and deterministic:
                    for i in range(len_option):
                        sOpt, rOpt = stateOption[i], rewardOption[i]
                        node.add_evaluator_transition(sOpt, local_action_evaluator, rOpt, nextstate, real_done, node, goal_id, id, next_id, penality = PENALITY)

                if inOption and args.final_trajectory:
                    node.add_external_transition(batch_state, batch_action, batch_nextstate, next_id)


                observation = env.reset()
                observation = torch.tensor(observation).type(torch.float)
                print(seed, ":", e, "/", n_epoch, "  >REWARD:", r)
                r=0
                time_step = 0
                inOption, rewardOption, stateOption, len_option = False, np.array([]), [], 0
                batch_state, batch_nextstate, batch_action = [], [], []

                lms.optimize_agents_external(N_TRAIN[0])
                lms.optimize_agents_inside(N_TRAIN[1])
                lms.optimize_evaluator(N_TRAIN[2])

                if deterministic and args.determinist:
                    deterministic = False
                else:
                    deterministic = True

            if e%freq_val==0:
                current_reward, current_list_reward = evaluate_agent(env_test, lms, n_starts=5, render = False)

                print(seed, ":", "VALIDATION:", current_reward)
                reward_list += current_list_reward
                step += [e for _ in range(len(current_list_reward))]
                use = [inside, external, tot]

                print(PATH_SAVE+"/rewards_"+num+".txt")
                with open(PATH_SAVE+"/rewards_"+num+".txt", 'w') as fp:
                   json.dump([reward_list, step, use], fp)

                print("\n##################################")
                print("Information about buffers:")
                current_iter_buffer = [True, True, True]
                for n in lms.modules_list:
                    [eval_buff, externe_buff, internal_buff] = n.print_info()
                    current_iter_buffer[0] = current_iter_buffer[0] and (eval_buff>=buffer_size_goal[0])
                    current_iter_buffer[1] = current_iter_buffer[1] and (externe_buff>=buffer_size_goal[1])
                    current_iter_buffer[2] = current_iter_buffer[2] and (internal_buff>=buffer_size_goal[2])

                test_buffer_size[0] = test_buffer_size[0] or current_iter_buffer[0]
                test_buffer_size[1] = test_buffer_size[1] or current_iter_buffer[1]
                test_buffer_size[2] = test_buffer_size[2] or current_iter_buffer[2]
                for i in range(3):
                    if test_buffer_size[i] and info_buffers[i]==-1:
                        info_buffers[i]=e
                print("First time all buffer are filled: ", info_buffers)
                print("##################################\n")
                if current_reward >= best:
                    best = current_reward
                    print(">> BEST SCORE")
                    lms.save_model(file = PATH_SAVE)



        print("\n##################################")
        print("Information about buffers:")
        print("\n##################################")
        print("Information about buffers:")
        current_iter_buffer = [True, True, True]
        for n in lms.modules_list:
            [eval_buff, externe_buff, internal_buff] = n.print_info()
            current_iter_buffer[0] = current_iter_buffer[0] and (eval_buff>=buffer_size_goal[0])
            current_iter_buffer[1] = current_iter_buffer[1] and (externe_buff>=buffer_size_goal[1])
            current_iter_buffer[2] = current_iter_buffer[2] and (internal_buff>=buffer_size_goal[2])

        test_buffer_size[0] = test_buffer_size[0] or current_iter_buffer[0]
        test_buffer_size[1] = test_buffer_size[1] or current_iter_buffer[1]
        test_buffer_size[2] = test_buffer_size[2] or current_iter_buffer[2]

        for i in range(3):
            if test_buffer_size[i] and info_buffers[i]==-1:
                info_buffers[i]=e
        print("First time all buffer are filled: ", info_buffers)
        print("##################################\n")
        #lms.save_model(file = PATH_SAVE)
