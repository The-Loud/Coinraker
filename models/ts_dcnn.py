"""
The multi-channel DCNN will take in each time series as a separate channel.
These channels will be convolved separately with independent filters.
Once completed, the model will concatenate all the feature maps to a linear layer
and pass it through some hidden layers for a final regression.
# Kind of like this paper: http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf
"""
from torch import nn


def linear_layer(inp: int, outp: int) -> nn.Sequential:
    """
    Linear layer definition that uses two hidden layers
    :param inp: Size of the starting input layer. This is derived from the third dimension.
    :return: nn.Sequential completed block
    """
    return nn.Sequential(
        nn.Linear(inp, 50),
        nn.ReLU(inplace=True),
        nn.Linear(50, 25),
        nn.ReLU(inplace=True),
        nn.Linear(25, 1),
    )


class BitNet(nn.Module):
    """
    The main network.
    One convNet layer and one linear layer.
    """

    def __init__(self):
        super().__init__()

        # Need to figure out how to get this value from the output of the CNNs
        self.lin1 = linear_layer(75, 1)

    def forward(self, x_data):
        """
        Each tensor will have a couple of channels.
        Each channel should be sent through its own CNN
        [Batch, channel, subsequence]
        :param x_data: input tensor
        :return: vector ready for linear layer
        """
        out = self.lin1(x_data)
        return out.flatten()
