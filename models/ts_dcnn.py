"""
The multi-channel DCNN will take in each time series as a separate channel.
These channels will be convolved separately with independent filters.
Once completed, the model will concatenate all the feature maps to a linear layer
and pass it through some hidden layers for a final regression.
# Kind of like this paper: http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf
"""
from torch import nn


class BitNet(nn.Module):
    """
    Linear regression model with PyTorch
    """

    def __init__(self):
        super().__init__()

        # Need to figure out how to get this value from the output of the CNNs
        self.lin1 = nn.Linear(5, 1)

    def forward(self, x_data):
        """passes the data through a single learnable layer."""
        out = self.lin1(x_data)
        return out
