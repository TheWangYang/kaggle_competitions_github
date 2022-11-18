import torch
import torch.nn as nn


# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.conv = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),

            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
            nn.ReLU(inplace=True),

            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.ReLU(inplace=True),

            # 第五层卷积
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, img):
        x = self.conv(img)
        x = torch.flatten(x, 1)
        x = self.fc(x.view(x.shape[0], -1))
        return x




