import torch
import torch.nn as nn
from typing import Any


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 4) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            SahiConv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 2)),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            SahiConv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 2)),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            SahiConv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SahiConv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SahiConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((3, 2)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    if pretrained:
        pass
    return model


if __name__ == "__main__":
    x = torch.zeros(1, 3, 224, 224)
    net = alexnet(pretrained=False)
    y = net(x)
    print(y.shape)


