import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from utils import split_sequence


# Build a multi-channel ConvNet
# Kind of like this paper: http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf

'''
The multi-channel DCNN will take in each time series as a separate channel.
This model differs from the MC-DCNN in that each channel is combined and passed over by the kernel once.
Once completed, the model will concatenate all the feature maps to a linear layer
and pass it through some hidden layers for a final regression.
'''

def conv_1d(inp: int, oup: int, k_size: tuple[int, ...], stride: tuple[int, ...],
            padding: tuple[int, ...]) -> nn.Sequential:
    """
    Creates a standard convolutional block with batchnorm and activation.
    :param inp: input channels. Typically 1 for time series
    :param oup: number of output channels
    :param k_size: kernel size. 1 x k
    :param stride: self-explanatory
    :param padding: self-explanatory
    :return: complete conv block
    """
    return nn.Sequential(
        nn.Conv1d(inp, oup, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(oup),
        nn.ReLU(inplace=True),
    )

def linear(inp: int):
    pass



class SigNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(SigNet, self).__init__()

        self.block1 = conv_1d(inp=in_channels, oup=8, k_size=(3,), stride=(1,), padding=(1,))  # inp x 8 x 24
        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=False)  # inp x 8 x 12
        self.block2 = conv_1d(8, 16, (6,), (1,), (1,))  # 1 x 16 x 7

    def forward(self, x):
        x_1 = self.pool(self.block1(x))

        # TODO: Verify if a second pool is necessary
        x_2 = self.pool(self.block2(x_1))

        # Reshape tensors for the concat
        x_1 = x_1.reshape(-1, 1, 1)
        x_2 = x_2.reshape(-1, 1, 1)

        out = torch.cat((x_1, x_2)).reshape(1, 1, -1)  # 160
        return out


class BitNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(BitNet, self).__init__()

        self.block1 = SigNet(in_channels)


        # Create linear layers here.
        self.lin1 = nn.Linear(160, 500)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(500, 50)
        self.lin3 = nn.Linear(50, 1)

    def forward(self, x):
        out = self.block1(x)

        # Linear stuff
        out = self.relu(self.lin1(out))
        out = self.relu(self.lin2(out))

        # We want a linear output
        out = self.lin3(out)
        return out

