import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import time
import pandas as pd

data = {"data_sac": {}, "data_lms": {}, "data_qp": {}}
fig, axes = plt.subplots(nrows=2, ncols=1)
window_size = 8
nb_test = 5
max_iter = 44400 #91800
use_sac, use_lms, use_qp = True, True, True

list_sac = [0, 1, 2, 5, 3, 4, 6, 7]
list_lms = list_sac
list_qp = [0, 1, 2, 3, 4, 5, 6, 7] #list_qp

axes[0].set_title("Rolling Mean (size = "+ str(window_size)+")")
axes[1].set_title("Rolling Std (size = "+ str(window_size)+")")

# axes[0].set_xlim(0, max_iter)
# axes[1].set_xlim(0, max_iter)

if use_sac:
    nb_sample_sac = len(list_sac)
    nb_it = []
    for i in range(len(list_sac)):
        print(">> Load: ./Results/CartStemContact/sac/rewards_"+ str(list_sac[i]) + ".txt")
        new_reward, new_iter = [], []
        with open("./Results/CartStemContact/sac/rewards_"+ str(list_sac[i]) + ".txt", 'r') as fp:
            loaded_data = json.load(fp)
            nb_it.append(loaded_data[1][-1])
            for j in range(len(loaded_data[1])):
                if not (loaded_data[1][j] in data["data_sac"]):
                    data["data_sac"][loaded_data[1][j]] = loaded_data[0][j]/nb_test
                else:
                    data["data_sac"][loaded_data[1][j]] += loaded_data[0][j]/nb_test
    nb_it_min = min(nb_it)
    print(">>      NB MIN (SAC):", nb_it_min, nb_it)
    new_data_sac = {}
    for key in data["data_sac"].keys():
        if key <= nb_it_min:
            new_data_sac[key] = data["data_sac"][key]/nb_sample_sac
    data["data_sac"] = new_data_sac

if use_lms:
    nb_sample_lms = len(list_lms)
    nb_it = []
    for i in range(len(list_lms)):
        print(">> Load: ./Results/CartStemContact/meta_states/rewards_"+ str(list_lms[i]) + ".txt")
        with open("./Results/CartStemContact/meta_states/rewards_"+ str(list_lms[i]) + ".txt", 'r') as fp:
            loaded_data = json.load(fp)
            nb_it.append(loaded_data[1][-1])
            for j in range(len(loaded_data[1])):
                if not (loaded_data[1][j] in data["data_lms"]):
                    data["data_lms"][loaded_data[1][j]] = loaded_data[0][j]/nb_test
                else:
                    data["data_lms"][loaded_data[1][j]] += loaded_data[0][j]/nb_test
    nb_it_min = min(nb_it)
    print(">>      NB MIN (LMS):", nb_it_min, nb_it)
    new_data_lms = {}
    for key in data["data_lms"].keys():
        if key <= nb_it_min:
            new_data_lms[key] = data["data_lms"][key]/nb_sample_lms
    data["data_lms"] = new_data_lms

if use_qp:
    nb_sample_qp = len(list_qp)
    nb_it = []
    for i in range(len(list_qp)):
        print(">> Load: ./Results/CartStemContact/QP/rewards_"+ str(list_qp[i]) + ".txt")
        with open("./Results/CartStemContact/QP/rewards_"+ str(list_qp[i]) + ".txt", 'r') as fp:
            loaded_data = json.load(fp)
            nb_it.append(loaded_data[1][-1])
            for j in range(len(loaded_data[1])):
                if not (loaded_data[1][j] in data["data_qp"]):
                    data["data_qp"][loaded_data[1][j]] = loaded_data[0][j]/nb_test
                else:
                    data["data_qp"][loaded_data[1][j]] += loaded_data[0][j]/nb_test
    nb_it_min = min(nb_it)
    print(">>      NB MIN (QP):", nb_it_min, nb_it)
    new_data_qp = {}
    for key in data["data_qp"].keys():
        if key <= nb_it_min:
            new_data_qp[key] = data["data_qp"][key]/nb_sample_qp
    data["data_qp"] = new_data_qp

pdData_base = pd.DataFrame.from_dict(data)
legend = []
if use_lms:
    pdData_base["lms"] = pdData_base["data_lms"].rolling(window=window_size, center=False).mean()
    pdData_base["lms_std"] = pdData_base["data_lms"].rolling(window=window_size, center=False).std()
    pdData_base["lms"].plot(ax=axes[0], color = "blue")
    pdData_base["lms_std"].plot(ax=axes[1], color = "blue")
    legend.append("Meta-states")

if use_sac:
    pdData_base["sac"] = pdData_base["data_sac"].rolling(window=window_size, center=False).mean()
    pdData_base["sac_std"] = pdData_base["data_sac"].rolling(window=window_size, center=False).std()
    pdData_base["sac"].plot(ax=axes[0], color = "orange")
    pdData_base["sac_std"].plot(ax=axes[1], color = "orange")
    legend.append("SAC")

if use_qp:
    pdData_base["qp"] = pdData_base["data_qp"].rolling(window=window_size, center=False).mean()
    pdData_base["qp_std"] = pdData_base["data_qp"].rolling(window=window_size, center=False).std()
    pdData_base["qp"].plot(ax=axes[0], color = "purple")
    pdData_base["qp_std"].plot(ax=axes[1], color = "purple")
    legend.append("Internal QP")


fig.legend(tuple(legend))

plt.show()
