# -*- coding: utf-8 -*-
"""
The CartStemContactEnv is defined by several informations:
- posCart: the x position of the cart
- psTip: the x position of the tips
- poscontact: the x position of left contact, the x position of the right contact, the 3D dimension of the contact (/ center of the contact)
- goal: the x position of the goal

The stem is in contact if:
- leftcontact: state[0]< state[2] + 0.5*np.sqrt((2*state[4])**2 + (2*state[6])**2)
- rightcontact: state[0] > state[3] - 0.5*np.sqrt((2*state[4])**2 + (2*state[6])**2)

The robot is control is a continuous action of dim 1 (the position of the cart)

"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 16 2021"

import gym
import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import sofagym

USEBAD = False
USEV = True

os.makedirs("./Data/CartStemContact", exist_ok = True)

n_state = 3
paths_data = []
for i in range(n_state):
    name_path = "./Data/CartStemContact/state_"+str(i)

    name_data = name_path + "/data.txt"
    paths_data.append(name_data)
    os.makedirs(name_path, exist_ok = True)


env = gym.make("cartstemcontact-v2")
observation = env.reset()

data = [[] for _ in range(n_state)]
for i in range(n_state):
    with open(paths_data[i], 'r') as outfile:
        d = json.load(outfile)
        data[i] = d

go_to = 0
# ACTION = -1
for e in tqdm(range(500000)):
    save_elt = observation.tolist()

    if observation[0]<= observation[2] + 0.5*np.sqrt((2*observation[4])**2 + (2*observation[6])**2):
        num = 0
        # print("LEFT")
    elif observation[0] >= observation[3] - 0.5*np.sqrt((2*observation[4])**2 + (2*observation[6])**2):
        num = 2
        # print("RIGHT")
    else:
        num = 1
        # print("NO")

    data[num].append(save_elt)

    try:
        if go_to%30 == 0:
            action = [float(np.random.choice([-1, 1]))]
            go_to+=1
        elif go_to%30 in [1, 2, 3, 4, 5, 6, 7, 8]:
            go_to+=1
        else:
            action = [float(2*np.random.random()-1)]
            go_to+=1

        observation, reward, done, info = env.step(action)
        # env.render()

        if done:
            observation = env.reset()
            go_to = 0

            for i in range(n_state):
                with open(paths_data[i], 'w') as outfile:
                    json.dump(data[i], outfile)

            print(" ####   INFORMATION   ####  ")
            for i in range(n_state):
                print(">>  dataset ", i, ":", len(data[i]))
            print(" #########################  ")

    except:
        env.close()
        env = gym.make("cartstemcontact-v0")
        observation = env.reset()
        go_to = 0


    # if (e+1)%100000==0:
    #     if ACTION == -1:
    #         ACTION = 1
    #     else:
    #         ACTION = -1



env.close()
