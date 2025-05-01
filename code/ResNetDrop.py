import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockDrop(nn.Module):
  def __init__(self, survival_rate, in_channels, out_channels, dropout, stride=1):
    super(ResidualBlockDrop, self).__init__()
    self.survival_rate = survival_rate
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.dropout = dropout
    self.stride = stride

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(inplace=True)
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

    if self.training:
      gate = torch.rand(1).item() < self.survival_rate
      if gate: #do not skip if gate is open (1)
        output += residual
      else: #skip 
        output = residual
    else:
      output = self.survival_rate * output + residual
  
    output = F.relu(output)
    return output

class ResNetDrop(nn.Module):
  def __init__(self, num_blocks, num_classes=10):
    super(ResNetDrop, self).__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias= False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
    )

    #find the survival probabilities by lineat decay
    survival_rates = []
    L = num_blocks*3
    rate_L = 0.5
    for l in range(1, L+1):
      rate_l = 1 - (l/L) * (1 - rate_L)
      survival_rates.append(rate_l)

    survival_rates1 = survival_rates[:num_blocks]
    survival_rates2 = survival_rates[num_blocks: 2*num_blocks]
    survival_rates3 = survival_rates[2*num_blocks: 3*num_blocks]
    self.layer1 = self.block_layer(num_blocks, survival_rates1, 16, 16, stride=1)
    self.layer2 = self.block_layer(num_blocks, survival_rates2, 16, 32, stride=2)
    self.layer3 = self.block_layer(num_blocks, survival_rates3, 32, 64, stride=2)

    self.last = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
        )


  def block_layer(self, num_blocks, survival_rates, in_channel, out_channel, stride):
    layers = []
    for i in range(num_blocks):
      layer = ResidualBlockDrop(survival_rates[i], in_channel, out_channel, dropout=0, stride=stride if i == 0 else 1)
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