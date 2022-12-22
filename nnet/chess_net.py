# net used to train model. As a general rule, the default settings for convolution layers, residual layers have been used.
from torch import nn
import torch.nn.functional as F

from settings import NUMBER_OF_RES_LAYERS


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 64*73
        self.conv1 = nn.Conv2d(119, 256, 3)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        # batch size * board_features * board_squares
        identity = s.view(-1, 119, 64)
        out = self.conv1(s)
        out = self.bn1(out)
        s = F.relu(out)
        return s


class ResBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, down_sample=None):
        super(ResBlock, self).__init__()
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample

        out += identity
        out = F.relu(out)

        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # value head
        self.conv1_v = nn.Conv2d(256, 1, 1)
        self.bn1_v = nn.BatchNorm2d(1)
        self.fc1_v = nn.Linear(64, 64)
        self.fc2_v = nn.Linear(64, 1)

        # policy head
        self.conv1_p = nn.Conv2d(256, 2, 1)
        self.bn1_p = nn.BatchNorm2d(2)
        self.fc1_p = nn.Linear(128, 1)

    def forward(self, x):
        # value head
        out_v = x
        out_v = self.conv1_v(out_v)
        out_v = self.bn1_v(out_v)
        out_v = F.relu(out_v)
        out_v = out_v.view(-1, 64)  # batch size * number of squares
        out_v = self.fc1_v(out_v)
        out_v = F.relu(out_v)
        out_v = self.fc2_v(out_v)
        out_v = F.tanh(out_v)

        # policy head
        out_p = x
        out_p = self.conv1_p(out_p)
        out_p = self.bn1_p(out_p)
        out_p = F.relu(out_p)
        out_p = out_p.view(-1, 64*128)
        out_p = self.fc1_p(out_p)

        return out_v, out_p


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for i in range(NUMBER_OF_RES_LAYERS):
            setattr(self, f"res_{i+1}", ResBlock())
        self.head = OutBlock()

    def forward(self, s):
        out = self.conv(s)
        for i in range(NUMBER_OF_RES_LAYERS):
            out = getattr(self, f"res_{i+1}")(out)
        out = self.head(out)
        return out
