import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, dropout, stride=1):
    super(ResidualBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.dropout = dropout
    self.stride = stride

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels)
    )


    skip_layers = [nn.Identity()]
    if stride > 1:
      skip_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
    self.skip = nn.Sequential(*skip_layers)

    self.need_padding = out_channels > in_channels
    self.padding = out_channels - in_channels if self.need_padding else 0

  def forward(self, x):
    output = self.net(x)
    residual = self.skip(x)
    if self.need_padding:
      pad = torch.zeros(residual.size(0), self.padding, residual.size(2), residual.size(3),
                            device=residual.device, dtype=residual.dtype)
      residual = torch.cat([residual, pad], dim=1)

    output += residual
    output = F.relu(output)

    return output

class ResNet(nn.Module):
  def __init__(self, num_blocks, num_classes=10):
    super(ResNet, self).__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias= False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
    )

    self.layer1 = self.block_layer(num_blocks, 16, 16, stride=1)
    self.layer2 = self.block_layer(num_blocks, 16, 32, stride=2)
    self.layer3 = self.block_layer(num_blocks, 32, 64, stride=2)

    self.last = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
        )


  def block_layer(self, num_blocks, in_channel, out_channel, stride):
    layers = []
    for i in range(num_blocks):
      layer = ResidualBlock(in_channel, out_channel, dropout=0, stride=stride if i == 0 else 1)
      layers.append(layer)
      in_channel = out_channel
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.first(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.last(x)

    return x