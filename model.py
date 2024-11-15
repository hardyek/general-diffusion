import torch
import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    def __init__(self, c):
        super(block, self).__init__()
        self.conv_1 = nn.Conv2d(c,c, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(c,c, kernel_size=3, padding=1)

    def forward(self, x):
        res = x

        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)

        x += res

        x = F.relu(x)
        return x

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.middle = nn.ModuleList(
            [block(128) for _ in range(3)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)

        for block in self.middle:
            x = block(x)

        x = self.decoder(x)
        return x
