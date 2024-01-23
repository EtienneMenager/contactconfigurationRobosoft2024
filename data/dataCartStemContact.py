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
import random
from PIL import Image

def get_data_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class CartStemContactDatasetState(Dataset):
    def __init__(self, path = "./Data", n_state = 16, validation = False):
        super(CartStemContactDatasetState, self).__init__()
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
                print(">> Len data", i, ":", len_data[i])

        l_data = min(len_data)
        for i in range(self.n_state):
            d = data[i]
            random.shuffle(d)
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
