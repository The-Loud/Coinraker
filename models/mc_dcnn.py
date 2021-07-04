import torch
import torch.nn as nn
from typing import List

# Build a multi-channel ConvNet
# Kind of like this paper: http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf

'''
The multi-channel DCNN will take in each time series as a separate channel.
These channels will be convolved separately with independent filters.
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


def linear_layer(inp: int):
    return nn.Sequential(
        nn.Linear(inp, 500),
        nn.ReLU(inplace=True),
        nn.Linear(500, 50),
        nn.Linear(50, 1),
    )


class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()

        self.block1 = conv_1d(inp=1, oup=8, k_size=(3,), stride=(1,), padding=(1,))  # 1 x 8 x 24
        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=False)  # 1 x 8 x 12
        self.block2 = conv_1d(8, 16, (6,), (1,), (1,))  # 1 x 16 x 7

    def forward(self, x):
        x_1 = self.pool(self.block1(x))

        # TODO: Verify if a second pool is necessary
        x_2 = self.pool(self.block2(x_1))

        # Reshape tensors for the concat
        x_1 = x_1.reshape(1, 1, -1)
        x_2 = x_2.reshape(1, 1, -1)

        out = torch.cat((x_1, x_2), dim=2)  # 160
        return out


class BitNet(nn.Module):
    def __init__(self, series: int = 1):
        super(BitNet, self).__init__()

        # Create a SigNet for each channel
        self.series = series  # number of time-series channels (features)
        self.convs = nn.ModuleList()
        for i in range(self.series):
            self.convs.append(SigNet())

        # TODO: make a tensor of the appropriate dimensions and concat the outputs
        some_tensor = torch.empty([1, 1, 160])

        # Loop through each feature vector and ConvNet
        # TODO: Should this be in the forward method?
        '''for s, c in zip(self.series, self.convs):
            output = c(s)
            torch.cat((some_tensor, output))

        self.block1 = conv_1d(inp=1, oup=8, k_size=(3,), stride=(1,), padding=(1,))  # 1 x 8 x 24
        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=False)  # 1 x 8 x 12
        self.block2 = conv_1d(8, 16, (6,), (1,), (1,))  # 1 x 16 x 7 or 9 if ceil_mode = True

        # Create linear layers here.
        self.lin1 = nn.Linear(some_tensor.shape[3], 500)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(500, 50)
        self.lin3 = nn.Linear(50, 1)'''

        # Need to figure out how to get this value from the output of the CNNs
        self.lin = linear_layer(some_tensor.shape[2])

    def forward(self, x):

        out = torch.tensor([1, 1, x.shape[2]])
        # TODO: Should this be in the forward method?
        # Each tensor will have a couple of channels. Each channel should be sent through its own CNN
        # [Batch, channel, subsequence]
        for i in range(x.shape[1]):
            output = self.convs[i](x[:, i, :])
            torch.cat((out, output), dim=2)

        # Linear stuff
        out = self.lin(out)

        return out

