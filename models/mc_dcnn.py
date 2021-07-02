import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time


# Build a multi-channel ConvNet
# Kind of like this paper: http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf

'''
The multi-channel DCNN will take in each time series as a separate channel.
These channels will be convoled separately with independent filters.
Once completed, the model will concatenate all the feature maps to a linear layer
and pass it through some hidden layers for a final regression.
'''


# Make a simple function to build the conv blocks
def conv_1d(inp, oup, k_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(inp, oup, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(oup),
        nn.ReLU(inplace=True),
    )


class BitNet(nn.Module):
    def __init__(self):
        super(BitNet, self).__init__()

        self.block1 = conv_1d(inp=1, oup=8, k_size=3, stride=1, padding=0)  # one in-channel, 8 out channels and kernel size 3
        self.pool = nn.MaxPool1d(kernel_size=2)  # halve the input size
        self.block2 = conv_1d(8, 16, 6, 1, 0)  # gotta make sure of the sizes here

        # Create linear layers here. Need to determine the size

    def forward(self, x):
        x = self.block1(x)
        x_1 = self.pool(x)
        x_2 = self.pool(self.block2(x_1))

        # Pull together the feature maps from earlier conv layers
        x = torch.cat(x_1, x_2)

