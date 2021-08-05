"""
The multi-channel DCNN will take in each time series as a separate channel.
These channels will be convolved separately with independent filters.
Once completed, the model will concatenate all the feature maps to a linear layer
and pass it through some hidden layers for a final regression.
# Kind of like this paper: http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf
"""
import torch
from torch import nn


def conv_1d(
    inp: int,
    oup: int,
    k_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
) -> nn.Sequential:
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
        nn.Conv1d(
            inp, oup, kernel_size=k_size, stride=stride, padding=padding, bias=True
        ),
        nn.BatchNorm1d(oup),
        nn.ReLU(inplace=True),
    )


def linear_layer(inp: int, oup: int) -> nn.Sequential:
    """
    Linear layer definition that uses two hidden layers
    :param inp: Size of the starting input layer. This is derived from the third dimension.
    :return: nn.Sequential completed block
    """
    return nn.Sequential(nn.Linear(inp, oup))


class SigNet(nn.Module):
    """
    This is the neural network for the Convolutional layers.
    This method allows for dynamic generation of CNNs in which each feature layer
    is concatenated and flattened at the output layer for input
    to the linear layer.

    Two layers with a 1x3 kernel and then a 1x6 kernel.
    The idea is to capture different subsequences for better predictions.
    """

    def __init__(self):
        super().__init__()

        self.block1 = conv_1d(inp=5, oup=10, k_size=(2,), stride=(1,), padding=(1,))
        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        self.block2 = conv_1d(8, 16, (2,), (1,), (1,))
        # self.block3 = conv_1d(16, 16, (6,), (1,), (1,))
        # self.block4 = conv_1d(16, 32, (3,), (1,), (1,))

    def forward(self, x_inp):
        """
        Forward method to pass data through each pass
        :param x_inp: tensor
        :return: prediction
        """
        x_1 = self.block1(x_inp)
        # x_1 = self.block1(x_inp)

        # No need for second pool
        # x_2 = (self.block2(x_1))
        # print(x_2.shape)

        # x_3 = self.block3(x_2)
        # x_4 = self.block4(x_3)

        # Reshape tensors for the concat
        # x_1 = x_1.reshape(1, 1, -1)
        # x_2 = x_2.reshape(1, 1, -1)
        # x_3 = x_3.reshape(1, 1, -1)
        # x_4 = x_4.reshape(1, 1, -1)

        # TODO: Test with a residual block here
        # out = torch.cat((x_1, x_2, x_3, x_4), dim=2)
        return x_1.flatten()


class BitNet(nn.Module):
    """
    The main network.
    A dynamic number of ConvNet layers is initialized depending on the number
    of channels (time-series or features).
    After each is set up, the model will pass each time-series through its own
    separate CNN. The idea is to have the network view the time-series independently
    and assess their separate impact on the dependent variable. A single multi-channel
    CNN would aggregate all the time-series data together in the first pass, assessing
    all the data at once.
    The data from each CNN is then concatenated to a 1D tensor and passed through a
    linear block.
    """

    def __init__(self):
        super().__init__()

        # Create a SigNet for each channel
        self.conv = SigNet()

        h_layer = 250

        # Need to figure out how to get this value from the output of the CNNs
        self.lin1 = linear_layer(h_layer, 1)

    def forward(self, x_data):
        """
        Each tensor will have a couple of channels.
        Each channel should be sent through its own CNN
        [Batch, channel, subsequence]
        :param x_data: input tensor
        :return: vector ready for linear layer
        """

        out = self.conv(x_data)

        # Linear stuff
        out = self.lin1(out)

        return out.flatten()
