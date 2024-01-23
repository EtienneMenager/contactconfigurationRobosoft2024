# -*- coding: utf-8 -*-
"""Toolbox.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"

import torch
import torch.nn as nn
import numpy as np

class BlockLSTM(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(BlockLSTM, self).__init__()
        self.lstm = nn.LSTMCell(dim_in, dim_hidden)
    def forward(self, input, h, c):
        hnext, cnext = self.lstm(input, (h, c))
        return hnext, cnext


if __name__ == '__main__':
    dim_in = 4
    dim_hidden = 8

    h, c = torch.randn(1, dim_hidden), torch.randn(1, dim_hidden)
    input = torch.randn(1, dim_in)
    LSTM = BlockLSTM(dim_in, dim_hidden)
    hnext, cnext = LSTM(input, h, c)
    print(">>   LSTM")
    print(hnext, cnext)

    LL = LinearLayer(dim_in, dim_in)
    out = LL(input)
    l = nn.Linear(dim_in, dim_in)
    out_2 = l(input)
    print(">>   LL")
    print(out)
    print(">>   l")
    print(out_2)
