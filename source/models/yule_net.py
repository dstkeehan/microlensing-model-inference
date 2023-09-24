import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class YuleNet(nn.Module):
    def __init__(self, input_size = 720, output_size = 64):
        super(type(self), self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.cnn_1 = nn.Sequential(*[
            nn.Conv1d(1, 8, 64, 1, 32),
            nn.ReLU(),
            nn.AvgPool1d(16)
        ])

        self.cnn_2 = nn.Sequential(*[
            nn.Conv1d(8, 8, 64, 1, 32),
            nn.ReLU(),
            nn.AvgPool1d(int(self.input_size / 256))
        ])

        self.linear = nn.Sequential(*[
            nn.Dropout(0.5),
            nn.Linear(184, self.output_size),
            nn.ReLU()
        ])


    def forward(self, x):
        x = x.view(-1, 1, self.input_size)

        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = x.reshape(-1, 184)
        x = self.linear(x)

        return x


if __name__ == '__main__':
    model = YuleNet(720, 32)

    from torchinfo import summary

    summary(model, input_size = (1, 720), depth = 5, verbose = 1)