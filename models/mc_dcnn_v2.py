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
    """Creates a 1-dimensional convolution block with batch norm and ReLU.

    Args:
        inp (int): [description]
        oup (int): [description]
        k_size (tuple[int, ...]): [description]
        stride (tuple[int, ...]): [description]
        padding (tuple[int, ...]): [description]

    Returns:
        nn.Sequential: [description]
    """
    return nn.Sequential(
        # No bias term because batchnorm contains a bias term.
        nn.Conv1d(
            inp, oup, kernel_size=k_size, stride=stride, padding=padding, bias=False
        ),
        nn.BatchNorm1d(oup),
        nn.ReLU(inplace=True),
    )


def linear_layer(inp: int, oup: int) -> nn.Sequential:
    """Creates a standard linear layer.

    Args:
        inp (int): [description]
        oup (int): [description]

    Returns:
        nn.Sequential: [description]
    """
    return nn.Sequential(nn.Linear(inp, oup))


class SigNet(nn.Module):
    """
    This is the neural network for the Convolutional layers.
    This method allows for dynamic generation of CNNs in which each feature layer
    is concatenated and flattened at the output layer for input
    to the linear layer.

    Because the training data is so small, the model is greatly simplified.
    The idea is to capture different subsequences for better predictions.
    """

    def __init__(self):
        super().__init__()

        self.block1 = conv_1d(inp=1, oup=8, k_size=(5,), stride=(1,), padding=(1,))
        self.pool = nn.MaxPool1d(2)
        self.block2 = conv_1d(8, 16, 5, 1, 1)
        # self.block3 = conv_1d(64, 128, 3, 1, 1)

    def forward(self, x_inp):
        """
        Forward method to pass data through each pass
        :param x_inp: tensor
        :return: prediction
        """
        x_1 = self.block1(x_inp)
        x_1 = self.pool(x_1)
        x_2 = self.pool(self.block2(x_1))
        return x_2


class BitNet(nn.Module):
    """The main network.
        A dynamic number of ConvNet layers is initialized depending on the number
        of channels (time-series or features).
        After each is set up, the model will pass each time-series through its own
        separate CNN. The idea is to have the network view the time-series independently
        and assess their separate impact on the dependent variable. A single multi-channel
        CNN would aggregate all the time-series data together in the first pass, assessing
        all the data at once.
        The data from each CNN is then concatenated to a 1D tensor and passed through a
        linear block.

    Args:
        nn ([type]): [description]
    """

    def __init__(self, series: int = 1):
        super().__init__()

        # Create a SigNet for each channel
        self.series = series
        self.convs = nn.ModuleList()
        for _ in range(self.series):
            self.convs.append(SigNet())

        h_layer = 320

        # Need to figure out how to get this value from the output of the CNNs
        self.lin1 = linear_layer(h_layer, 1)

    def forward(self, x_data):
        """
        Each tensor will have a couple of channels.
        Each channel should be sent through its own CNN
        [Batch, channel, subsequence]
        :param x_data: input tensor
        :return: prediction
        """

        tensor_list = []
        for i in range(x_data.shape[1]):
            output = self.convs[i](x_data[:, i, :].unsqueeze(dim=1))
            tensor_list.append(output)

        out = torch.cat(tensor_list, dim=1)

        outputs = []
        for i in range(out.shape[0]):
            output = self.lin1(out[i, :, :].unsqueeze(0).reshape(1, 1, -1))
            outputs.append(output)

        out = torch.cat(outputs, dim=0)

        # Linear stuff
        # out = self.lin1(out)

        return out.flatten()
