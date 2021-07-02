import torch
import torch.nn as nn
from models.mc_dcnn import BitNet




net = BitNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.2)
loss_func = nn.MSELoss()