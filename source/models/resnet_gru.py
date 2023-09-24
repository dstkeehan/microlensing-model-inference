import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class ResnetGRU(nn.Module):
    def __init__(self, depth = 6, nlayer = 2, kernel_size = 9, hidden_conv = 2, max_hidden = 256, input_dim = 22, hidden_dim = 1, layer_dim = 2):
        super(type(self), self).__init__()

        network = list()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # Add the first residual block
        network.append(ResidualBlock(
            in_channels = 1, 
            out_channels = hidden_conv, 
            kernel_size = kernel_size))

        # Append more residual blocks until the block size has been reached
        for i in range(nlayer - 1):
            network.append(ResidualBlock(
                in_channels = hidden_conv, 
                out_channels = hidden_conv, 
                kernel_size = kernel_size))
        
        # Append new lots of blocks with MaxPool inbetween
        for i in range(depth - 1):
            dim_in = min(max_hidden, hidden_conv * 2 ** i)
            dim_out = min(max_hidden, hidden_conv * 2 ** (i + 1))

            # Add the maxpool layer between the blocks
            network.append(nn.MaxPool1d(
                kernel_size = 2, 
                stride = 2))

            # network.append(nn.Conv1d(
            #     in_channels=dim_in,
            #     out_channels=dim_in,
            #     kernel_size = 2, 
            #     stride = 2))

            # Append the correct number of next blocks
            for j in range(nlayer):
                network.append(ResidualBlock(
                    in_channels = dim_out if j != 0 else dim_in, 
                    out_channels = dim_out, 
                    kernel_size = kernel_size))
        
        self.resnet = nn.Sequential(*network)

        # This should be collating the hidden dimensions correctly
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first = True, bidirectional = False
        )

        # self.linear = nn.Linear(64 * 64, 128)

    def forward(self, x):
        # print(x.shape)
        # print(x.unsqueeze(1).shape)

        x = x.view(-1, 1, 720)

        x = self.resnet(x)

        # print(x.shape)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, _ = self.gru(x, h0.detach())

        out = out.reshape(-1, 64)

        # print(out.shape)

        # self.hidden = hidden

        # print(x.shape)
        # print(hidden.shape)

        # out = self.linear(out)

        # print(x.shape)

        return out



class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(type(self), self).__init__()

        # Define the network for each block
        self.network = nn.Sequential(*[
            # Normalised with a 1D convolution and padding
            weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding = int(kernel_size / 2))),
            nn.ReLU(),

            # Second sub-block
            weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding = int(kernel_size / 2))),
            nn.ReLU(),
        ])

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(type(self), self).__init__()

        # Block including two weight_norm convolutions with ReLU activation
        self.convolution = ConvolutionBlock(in_channels, out_channels, kernel_size)

        # If the number of channels in is not the same as the out channels, use a 1D convolution on residual connection
        self.residual_convolution = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Apply block to input data
        y = self.convolution(x)

        # If the number of input channels is not the same as the output, apply the transformation
        if self.residual_convolution is not None:
            return y + self.residual_convolution(x)

        return y + x


if __name__ == '__main__':
    model = ResnetGRU()

    from torchinfo import summary

    summary(model, input_size = (1, 720), depth = 5, verbose = 1)