# network.py
from typing import Type, Union
import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class ResidualBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RadarHPE3DNet(nn.Module):
    def __init__(
        self,
        expansion: int = 2,
        base_filters: int = 32,
        in_channels: int = 8,
        block: Union[Type[ResidualBlock], Type[Bottleneck]] = ResidualBlock,
        keypoints: int = 21,
    ):
        super(RadarHPE3DNet, self).__init__()
        self.keypoints = keypoints

        # Input projection
        c0 = base_filters * expansion
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c0,
            kernel_size=3,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(c0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        c1 = base_filters * expansion * block.expansion
        self.layer1 = block(inplanes=c0, planes=base_filters * expansion)
        self.layer2 = block(inplanes=c1, planes=base_filters * expansion)

        c2 = (base_filters * 2) * expansion * block.expansion
        self.layer3 = block(inplanes=c1, planes=(base_filters * 2) * expansion, stride=2)
        self.layer4 = block(inplanes=c2, planes=(base_filters * 2) * expansion)

        c3 = (base_filters * 2) * expansion * block.expansion
        self.layer5 = block(inplanes=c2, planes=(base_filters * 2) * expansion, stride=2)
        self.layer6 = block(inplanes=c3, planes=(base_filters * 2) * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        out_features = (base_filters * 2) * expansion * block.expansion
        self.landmarks = nn.Linear(out_features, keypoints * 3)
        self.hand_presence = nn.Sequential(
            nn.Linear(out_features, 1),
            nn.Sigmoid(),
        )
        self.handedness = nn.Sequential(
            nn.Linear(out_features, 1),
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        
        # Reshape to match label dimension: (batch, 21, 3)
        landmarks = self.landmarks(x).view(-1, self.keypoints, 3)
        hand_presence = self.hand_presence(x)
        handedness = self.handedness(x)

        return landmarks, hand_presence, handedness

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )