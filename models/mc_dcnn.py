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


def linear_layer(inp: int) -> nn.Sequential:
    """
    Linear layer definition that uses two hidden layers
    :param inp: Size of the starting input layer. This is derived from the third dimension.
    :return: nn.Sequential completed block
    """
    return nn.Sequential(nn.Linear(inp, 1))


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

        self.block1 = conv_1d(inp=1, oup=8, k_size=(3,), stride=(1,), padding=(1,))
        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=False)  # 1 x 8 x 12
        self.block2 = conv_1d(8, 16, (5,), (1,), (1,))  # 1 x 16 x 7

    def forward(self, x_inp):
        """
        Forward method to pass data through each pass
        :param x_inp: tensor
        :return: prediction
        """
        x_1 = self.pool(self.block1(x_inp))

        # x_2 = self.pool(self.block2(x_1))  # 1 x 8 x 24
        # No need for second pool
        x_2 = self.block2(x_1)  # 1 x 8 x 12
        # print(x_2.shape)

        # Reshape tensors for the concat
        x_1 = x_1.reshape(1, 1, -1)
        x_2 = x_2.reshape(1, 1, -1)

        # TODO: Test with a residual block here
        out = torch.cat((x_1, x_2), dim=2)
        return out


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

    def __init__(self, series: int = 1):
        super().__init__()

        # Create a SigNet for each channel
        self.series = series  # number of time-series channels (features)
        self.convs = nn.ModuleList()
        for _ in range(self.series):
            self.convs.append(SigNet())

        h_layer = 1280

        # Need to figure out how to get this value from the output of the CNNs
        self.lin = linear_layer(h_layer)

    def forward(self, x_data):
        """
        Each tensor will have a couple of channels.
        Each channel should be sent through its own CNN
        [Batch, channel, subsequence]
        :param x_data: input tensor
        :return: vector ready for linear layer
        """

        tensor_list = []
        for i in range(x_data.shape[1]):
            output = self.convs[i](x_data[:, i, :].unsqueeze(dim=1))
            tensor_list.append(output)

        out = torch.cat(tensor_list, dim=2)

        # Linear stuff
        out = self.lin(out)

        return out.flatten()
