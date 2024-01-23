import gym
from stable_baselines3 import SAC
from gym.wrappers import TimeLimit
import json
import argparse
import os
import numpy as np
import torch
import random

TRAIN, TEST = True, True
def load_environment(id, rank, timer_limit, seed = 0):
    def _init():
        __import__('sofagym')
        env = gym.make(id)
        env.seed(seed + rank)
        env = TimeLimit(env, max_episode_steps=timer_limit)
        #env.reset()
        return env

    return _init


def test(env, model, epoch, n_test=1, render = False):
    if render:
        env.config.update({"render":2})
        env.config.update({"visuQP":True})
    reward_sum = 0
    liste_reward = []
    final_reward = []
    for t in range(n_test):
        print("Start >> Epoch", epoch, "- Test", t)
        obs = env.reset()
        if render:
            env.render()
        rewards = []
        done = False
        id = 0
        while not done:
            action, _states = model.predict(obs, deterministic = True)
            obs, reward, done, info = env.step(action)
            if render:
                print("Test", t, "- Epoch ", id ,"- Took action: ", action, "- Got reward: ", reward, " (done = ", done, ")")
                env.render()
            rewards.append(reward)
            id+=1
        print("Done >> Test", t, "- Reward = ", rewards, "- Sum reward:", sum(rewards))
        reward_sum+= sum(rewards)
        liste_reward.append(sum(rewards))
        final_reward.append(reward)
    print("[INFO]  >> Mean reward: ", reward_sum/n_test)
    return reward_sum/n_test, liste_reward, final_reward


parser = argparse.ArgumentParser()

parser.add_argument("--n_epoch", "-ne", help="The number of epochs", type=int, default=100000)
parser.add_argument("--latent_size", "-hs", help="The size of hidden layers.", type=int, default = 512)
parser.add_argument("--batch_size", "-bs", help="The batch size.", type=int, default = 200)
parser.add_argument("--number_training", "-nt", help="The batch size.", type=int, default = 5)
parser.add_argument("--number_random", "-nr", help="The batch size.", type=int, default = 500)
parser.add_argument("--learning_rate", "-lr", help="The learning rate.", type=float, default = 1e-4)
parser.add_argument("--gammas", "-g", help="The gamma parameters.", type=float, default = 0.99)
parser.add_argument("-s", "--seed", help = "The seed",  type=int, required = True)
args = parser.parse_args()

gamma = args.gammas
learning_rate = args.learning_rate
batch_size = args.batch_size
number_training = args.number_training
number_random = args.number_random
size_layer = [args.latent_size for _ in range(3)]
n_epoch = args.n_epoch

id = "cartstemcontact-v2"
timer_limit = 30

env = load_environment(id, rank=0, timer_limit=timer_limit, seed=args.seed * 10)()
test_env = env

algo = 'SAC'
name = algo + "_" + id + "_" + str(args.seed * 10)
os.makedirs("./Results_benchmark/" + name, exist_ok=True)

if TRAIN:

    policy_kwargs = dict(net_arch=dict(pi=size_layer, qf=size_layer))
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=gamma, learning_rate=learning_rate,
                batch_size=batch_size, ent_coef='auto', learning_starts=number_random, gradient_steps = number_training)


    seed = args.seed * 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    reward_list, final_reward_list, steps = [], [], []
    best = -100000

    idx = 0
    print("\n-------------------------------")
    print(">>>    Start")
    print("-------------------------------\n")
    n_fail = 0
    time_scale = 200

    while idx < n_epoch:

        try:

            print("\n-------------------------------")
            print(">>>    Start training n°", idx + 1)
            print("[INFO]  >>    scene: ", id)
            print("[INFO]  >>    algo: ", algo)
            print("[INFO]  >>    seed: ", seed)
            print("[INFO]  >>    n_fail: ", n_fail)
            print("-------------------------------\n")

            model.learn(total_timesteps=timer_limit * time_scale, log_interval=20)
            model.save("./Results_benchmark/" + name + "/latest")

            print("\n-------------------------------")
            print(">>>    Start test n°", idx + 1)
            print("[INFO]  >>    scene: ", id)
            print("[INFO]  >>    algo: ", algo)
            print("[INFO]  >>    seed: ", seed)
            print("[INFO]  >>    n_fail: ", n_fail)
            print("-------------------------------\n")

            current_reward, current_list_reward, current_final_reward = test(test_env, model, idx, n_test=5)
            reward_list += current_list_reward
            final_reward_list += current_final_reward
            steps += [timer_limit * time_scale * (idx + 1) for _ in range(len(current_list_reward))]
            with open("./Results_benchmark/" + name + "/rewards_" + id + ".txt", 'w') as fp:
                json.dump([reward_list, steps, [], final_reward_list], fp)
            if current_reward >= best:
                print(">>>    Save training n°", idx + 1)
                model.save("./Results_benchmark/" + name + "/best")

            idx += 1
        except:
            print("[ERROR]  >> The simulation failed. Restart from previous id.")
            n_fail += 1

    model.save("./Results_benchmark/" + name + "/latest")

    print(">>   End.")

if TEST:
    save_path = "./Results_benchmark/" +  name + "/best"
    model = SAC.load(save_path)
    r, final_r = test(test_env, model, -1, n_test=5, render = True)
    print("[INFO]  >>    Best reward : ", r, " - Final reward:", final_r)