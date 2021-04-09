import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class CNN(nn.Module):
    def __init__(self, n_blocks, block, num_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(num_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layers1 = self._make_layer(block, 32, n_blocks[0])
        self.layers2 = self._make_layer(block, 64, n_blocks[1])
        self.layers3 = self._make_layer(block, 128, n_blocks[2])
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, n_blocks):
        layers = []
        for i in range(n_blocks):
            layers.append(block(self.in_planes, planes, 1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = F.avg_pool2d(out, out.shape[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def model(num_channels=3, num_classes=10, num_blocks=[1, 1, 1]):
    return CNN(n_blocks=num_blocks, block=BasicBlock, num_channels=num_channels, num_classes=num_classes)
