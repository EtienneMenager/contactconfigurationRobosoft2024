# -*- coding: utf-8 -*-
"""
The system is controlled by applying a force of +1 or -1
to the cart. The pendulum starts upright, and the goal is
to prevent it from falling over. A reward of +1 is provided
for every timestep that the pole remains upright. The episode
ends when the pole is more than 15 degrees from vertical,
or the cart moves more than 2.4 units from the center.

Description of CartPole:
Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf
Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right

Reward:
        Reward is 1 for every step taken, including the termination step

Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.

Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.

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

USEBAD = False
USEV = True

os.makedirs("./Data", exist_ok = True)

borne_inf, borne_sup = -0.3, 0.3
inter = [-0.3, 0, 0.3]
#inter = np.linspace(-0.3, 0.3, 5)
if not USEBAD and not USEV:
    n_state = len(inter)
else:
    n_state = 5 #3

paths_images = []
paths_data = []
for i in range(n_state-1):
    if not USEBAD and not USEV:
        name_path = "./Data/n_state="+str(n_state-1)+"/state_"+str(i)
    elif USEV:
        name_path = "./Data/v_n_state="+str(n_state-1)+"/state_"+str(i)
    else:
        name_path = "./Data/bad_n_state="+str(n_state-1)+"/state_"+str(i)

    name_img = name_path +"/img"
    name_data = name_path + "/data.txt"
    paths_images.append(name_img)
    paths_data.append(name_data)

    os.makedirs(name_path, exist_ok = True)
    os.makedirs(name_img, exist_ok = True)


env = gym.make("CartPole-v0")
observation = env.reset()

data = [[] for _ in range(n_state-1)]
n_images = [0 for _ in range(n_state-1)]

num_image = 30
L =1
dt = 0.027

stat = [[0 for _ in range(4)] for _ in range(4)]
for _ in tqdm(range(10000000)):

  save_elt = observation.tolist()

  if not USEBAD and not USEV:
      x, v, theta, omega = observation
      predicted_theta = theta + omega*dt
      predicted_x_cart = x + v*dt
      predicted_x_tip_pole = predicted_x_cart + np.sin(predicted_theta)*L


      diff = predicted_x_cart-predicted_x_tip_pole
      for i in range(n_state - 1):
          if diff > inter[i] and diff < inter[i+1]:
              data[i].append(save_elt)
              if n_images[i]< num_image:
                  img = env.render(mode="rgb_array")
                  img = Image.fromarray(img)
                  img.save(paths_images[i]+"/img_"+str(n_images[i])+".png")
                  n_images[i]+=1
  elif USEV and n_state == 3:
      x, v, theta, omega = observation
      diff = v + L*omega*np.cos(theta)

      if diff < 0:
          num = 0
      else:
          num = 1

      data[num].append(save_elt)
      if n_images[num]< num_image:
          img = env.render(mode="rgb_array")
          img = Image.fromarray(img)
          img.save(paths_images[num]+"/img_"+str(n_images[num])+".png")
          n_images[num]+=1


  elif USEV and n_state == 5:
      x, v, theta, omega = observation
      diff = v + L*omega*np.cos(theta)

      if diff < 0 and omega > 0:
          num = 0
      elif diff < 0 and omega <= 0:
          num = 1
      elif diff >= 0 and omega >= 0:
          num = 2
      else:
          num = 3

      data[num].append(save_elt)
      if n_images[num]< num_image:
          img = env.render(mode="rgb_array")
          img = Image.fromarray(img)
          img.save(paths_images[num]+"/img_"+str(n_images[num])+".png")
          n_images[num]+=1

  else:
      _, _, theta, _ = observation
      if theta < 0:
           data[0].append(save_elt)
           if n_images[0]< num_image:
                img = env.render(mode="rgb_array")
                img = Image.fromarray(img)
                img.save(paths_images[0]+"/img_"+str(n_images[0])+".png")
                n_images[0]+=1
      else:
          data[1].append(save_elt)
          if n_images[1]< num_image:
               img = env.render(mode="rgb_array")
               img = Image.fromarray(img)
               img.save(paths_images[1]+"/img_"+str(n_images[1])+".png")
               n_images[1]+=1


  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  x, v, theta, omega = observation.tolist()
  diff = v + L*omega*np.cos(theta)

  if diff < 0 and omega > 0:
      new_num = 0
  elif diff < 0 and omega <= 0:
      new_num = 1
  elif diff >= 0 and omega >= 0:
      new_num = 2
  else:
      new_num = 3

  stat[num][new_num]+=1
  if done:
    observation = env.reset()


for i in range(n_state - 1):
    with open(paths_data[i], 'w') as outfile:
        json.dump(data[i], outfile)

for i in range(4):
    for j in range(4):
        print("FROM ", i, " TO ", j, " : ", stat[i][j])
env.close()
