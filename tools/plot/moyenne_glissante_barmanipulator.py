import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import time
import pandas as pd

data = {"data_lms": {}, "data_sac": {}, "data_lms_follow": {}}
fig, axes = plt.subplots(nrows=2, ncols=1)
window_size = 8
max_iter = 300000
list_lms = [0, 1, 2, 3]
list_sac = [1]

use_num_reward = 0 #0: cumulative reward, 3: final reward
assert use_num_reward in [0, 3]
nb_test = 5
nb_state = 5
stat_node = None

use_lms, use_sac = True, False
axes[0].set_title("Rolling Mean (size = "+ str(window_size)+")")
axes[1].set_title("Rolling Std (size = "+ str(window_size)+")")

axes[0].set_xlim(0, max_iter)
axes[1].set_xlim(0, max_iter)

legend = []
if use_lms:
    nb_sample_lms = len(list_lms)
    nb_it = []
    for i in range(len(list_lms)):
        print(">> Load: ./Results/BarManipulator/meta_states/seed_"+ str(list_lms[i]*10) + "/rewards_"+ str(list_lms[i]) + ".txt")
        with open("./Results/BarManipulator/meta_states/seed_"+ str(list_lms[i]*10) + "/rewards_"+ str(list_lms[i]) + ".txt", 'r') as fp:
            loaded_data = json.load(fp)
            nb_it.append(loaded_data[1][-1])
            for j in range(len(loaded_data[1])):
                if not (loaded_data[1][j] in data["data_lms"]):
                    data["data_lms"][loaded_data[1][j]] = loaded_data[use_num_reward][j]/nb_test
                else:
                    data["data_lms"][loaded_data[1][j]] += loaded_data[use_num_reward][j]/nb_test
            print(">> Nb iterations:", nb_it[-1])
    nb_it_min = min(nb_it)
    new_data_lms = {}
    for key in data["data_lms"].keys():
        if key <= nb_it_min:
            new_data_lms[key] = data["data_lms"][key]/nb_sample_lms
    data["data_lms"] = new_data_lms

if use_sac:
    nb_sample_sac = len(list_lms)
    nb_it = []
    for i in range(len(list_sac)):
        print(">> Load: ./Results/BarManipulator/sac/rewards_"+ str(list_sac[i]) + ".txt")
        with open("./Results/BarManipulator/sac/rewards_"+ str(list_sac[i]) + ".txt", 'r') as fp:
            loaded_data = json.load(fp)
            nb_it.append(loaded_data[1][-1])
            for j in range(len(loaded_data[1])):
                if not (loaded_data[1][j] in data["data_sac"]):
                    data["data_sac"][loaded_data[1][j]] = loaded_data[use_num_reward][j]/nb_test
                else:
                    data["data_sac"][loaded_data[1][j]] += loaded_data[use_num_reward][j]/nb_test
            print(">> Nb iterations:", nb_it[-1])
    nb_it_min = min(nb_it)
    new_data_lms = {}
    for key in data["data_sac"].keys():
        if key <= nb_it_min:
            new_data_lms[key] = data["data_sac"][key]/nb_sample_sac
    data["data_sac"] = new_data_lms

pdData_base = pd.DataFrame.from_dict(data)
if use_lms:
    pdData_base["lms"] = pdData_base["data_lms"].rolling(window=window_size, center=False).mean()
    pdData_base["lms_std"] = pdData_base["data_lms"].rolling(window=window_size, center=False).std()
    pdData_base["lms"].plot(ax=axes[0], color = "orange")
    pdData_base["lms_std"].plot(ax=axes[1], color = "orange")
    legend.append("Meta-states")
if use_sac:
    pdData_base["sac"] = pdData_base["data_sac"].rolling(window=window_size, center=False).mean()
    pdData_base["sac_std"] = pdData_base["data_sac"].rolling(window=window_size, center=False).std()
    pdData_base["sac"].plot(ax=axes[0], color = "blue")
    pdData_base["sac_std"].plot(ax=axes[1], color = "blue")
    legend.append("SAC")


fig.legend(tuple(legend))

plt.show()
