# -*- coding: utf-8 -*-
"""Test the learning with CartPol example.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

from torch.utils.data import Dataset, DataLoader
import torch
import json
from tqdm import tqdm
import numpy as np
from PIL import Image

def get_data_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_images(path = "./Data", n_state = 10, n_image = 30):

    print(">>   Loading images ... ")
    images = []

    for i in tqdm(range(n_state)):
        path_img = path+"/state_"+str(i)+"/img"
        im_list = []
        for j in range(n_image):
            img = Image.open(path_img+"/img_"+str(i)+".png")
            img = np.array(img)
            im_list.append(img)

        images.append(im_list)

    return images

class CartPolDatasetStateOld(Dataset):
    def __init__(self, path = "./Data", validation = False):
        super(CartPolDatasetState, self).__init__()
        self.validation = validation

        self.gauche_path = path + "/gauche.txt"
        self.droite_path = path + "/droite.txt"
        self.centre_path = path + "/centre.txt"

        self.img_gauche_path = path + "/img_gauche.txt"
        self.img_droite_path = path + "/img_droite.txt"
        self.img_centre_path = path + "/img_centre.txt"

        self.load()

    def load(self):
        print(">>   Load data ...")

        with open("./Data/gauche.txt", 'r') as outfile:
            gauche = json.load(outfile)
        with open("./Data/droite.txt", 'r') as outfile:
            droite = json.load(outfile)
        with open("./Data/centre.txt", 'r') as outfile:
            centre = json.load(outfile)

        print(">>   Done ...")

        self.state = []
        self.id = []

        lg, ld, lc = len(gauche), len(droite), len(centre)
        if self.validation:
            start_g, end_g = int(lg*0.8), lg
            start_d, end_d = int(ld*0.8), ld
            start_c, end_c = int(lg*0.8), lc
        else:
            start_g, end_g = 0, int(lg*0.8)
            start_d, end_d = 0, int(ld*0.8)
            start_c, end_c = 0, int(lg*0.8)

        for i in tqdm(range(start_g, end_g)):
            self.state.append(torch.tensor(gauche[i]))
            self.id.append(0)

        for i in tqdm(range(start_d, end_d)):
            self.state.append(torch.tensor(droite[i]))
            self.id.append(1)

        for i in tqdm(range(start_c, end_c)):
            self.state.append(torch.tensor(centre[i]))
            self.id.append(2)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return self.state[idx], self.id[idx]

class CartPolDatasetState(Dataset):
    def __init__(self, path = "./Data", n_state = 16, validation = False):
        super(CartPolDatasetState, self).__init__()
        self.validation = validation
        self.n_state = n_state

        self.path = []
        for i in range(n_state):
            self.path.append(path+"/state_"+str(i))

        self.load()

    def load(self):
        print(">>   Load data ...")

        self.state = []
        self.id = []

        data, len_data = [], []
        for i in tqdm(range(self.n_state)):
            with open(self.path[i]+"/data.txt", 'r') as outfile:
                d = json.load(outfile)
                data.append(d)
                len_data.append(len(d))

        l_data = min(len_data)
        for i in range(self.n_state):
            d = data[i]
            if self.validation:
                start, end = int(l_data*0.8), l_data
                print("(validation) >> DATASET i = ", i, ":", l_data - int(l_data*0.8))
            else:
                start, end = 0, int(l_data*0.8)
                print("(train) >> DATASET i = ", i, ":", int(l_data*0.8))

            for j in range(start, end):
                self.state.append(torch.tensor(d[j]))
                self.id.append(i)


        print(">>   Done ...")


    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return self.state[idx], self.id[idx]

class CartPolDatasetTransition(Dataset):
    def __init__(self, path = "./Data", validation = False):
        super(CartPolDatasetTransition, self).__init__()
        self.transition = path + "/transition.txt"
        self.validation = validation
        self.load()

    def load(self):
        print(">>   Load data ...")

        with open("./Data/transition.txt", 'r') as outfile:
            transition = json.load(outfile)

        print(">>   Done ...")

        self.before = []
        self.action = []
        self.after = []

        for t in tqdm(transition):
            self.before.append(torch.tensor(t[0]))
            self.action.append(torch.tensor(t[1]))
            self.after.append(torch.tensor(t[2]))

        min_l = len(self.before)
        if self.validation:
            self.before, self.action, self.after = self.before[int(min_l*0.8):], self.action[int(min_l*0.8):], self.after[int(min_l*0.8):]
        else:
            self.before, self.action, self.after = self.before[:int(min_l*0.8)], self.action[:int(min_l*0.8)], self.after[:int(min_l*0.8)]

    def __len__(self):
        return len(self.before)

    def __getitem__(self, idx):
        return self.before[idx], self.action[idx], self.after[idx]
