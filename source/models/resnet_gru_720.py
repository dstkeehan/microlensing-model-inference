import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ConvolutionBlock(nn.Module):
    """Block containing convolutions and weighted normalisation for ResNet."""
    def __init__(self, in_channels, out_channels, kernel_size):
        """Create a convolution block containing the specified settings.

        Args:
            in_channels (int): number of input channels for convolution.
            out_channels (int): number of output channels for convolution.
            kernel_size (int): kernel size for convolutions.
        """
        super(type(self), self).__init__()

        # Define the network for each block
        self.network = nn.Sequential(
            *[
                # Normalised with a 1D convolution and padding
                weight_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=int(kernel_size / 2),
                    )
                ),
                nn.ReLU(),
                # Second sub-block
                weight_norm(
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        kernel_size,
                        padding=int(kernel_size / 2),
                    )
                ),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        """Apply forward network.

        Args:
            x (tensor): input data.

        Returns:
            out: network applied to input data.
        """
        return self.network(x)


class ResidualBlock(nn.Module):
    """Block containing residual connections and convolution networks for ResNet."""
    def __init__(self, in_channels, out_channels, kernel_size):
        """Create a residual block containing the specified settings.

        Args:
            in_channels (int): number of input channels for convolution.
            out_channels (int): number of output channels for convolution.
            kernel_size (int): kernel size for convolutions.
        """
        super(type(self), self).__init__()

        # Block including two weight_norm convolutions with ReLU activation
        self.convolution = ConvolutionBlock(in_channels, out_channels, kernel_size)

        # If the number of channels in is not the same as the out channels, use a 1D convolution on residual connection
        self.residual_convolution = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """Apply forward network.

        Args:
            x (tensor): input data.

        Returns:
            out: network applied to input data.
        """
        # Apply block to input data
        y = self.convolution(x)

        # If the number of input channels is not the same as the output, apply the transformation
        if self.residual_convolution is not None:
            return y + self.residual_convolution(x)

        return y + x

class ResnetGRU(nn.Module):
    """Class extending nn.Module from pytorch to define an embedding network for feature extraction on low-fidelity data."""

    def __init__(
        self,
        depth=8,
        nlayer=2,
        kernel_size=5,
        hidden_conv=4,
        max_hidden=256,
        input_dim=5,
        hidden_dim=1,
        layer_dim=2,
    ):
        """Create a Resnet + GRU featuriser network for input data of shape 1x720, with specified settings. This corresponds to low-fidelity microlensing light curves, which have 10x less observation cadence than is expected from ROMAN.

        Args:
            depth (int, optional): number of blocks for ResNet network. Defaults to 8.
            nlayer (int, optional): number of convolution layers in each ResNet block. Defaults to 2.
            kernel_size (int, optional): kernel size for convolution. Defaults to 5.
            hidden_conv (int, optional): starting expansion layers for input. Defaults to 4.
            max_hidden (int, optional): maximum size of hidden dimension. Defaults to 256.
            input_dim (int, optional): computed input dimension for GRU. Defaults to 5.
            hidden_dim (int, optional): hidden dimension for GRU. Defaults to 1.
            layer_dim (int, optional): number of layers for GRU. Defaults to 2.
        """
        super(type(self), self).__init__()

        network = list()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # Add the first residual block
        network.append(
            ResidualBlock(
                in_channels=1, out_channels=hidden_conv, kernel_size=kernel_size
            )
        )

        # Append more residual blocks until the block size has been reached
        for i in range(nlayer - 1):
            network.append(
                ResidualBlock(
                    in_channels=hidden_conv,
                    out_channels=hidden_conv,
                    kernel_size=kernel_size,
                )
            )

        # Append new lots of blocks with MaxPool inbetween
        for i in range(depth - 1):
            # Compute expansion of hidden dimension
            dim_in = min(max_hidden, hidden_conv * 2 ** i)
            dim_out = min(max_hidden, hidden_conv * 2 ** (i + 1))

            # Add the maxpool layer between the blocks
            network.append(nn.MaxPool1d(kernel_size=2, stride=2))

            # Append the correct number of next blocks
            for j in range(nlayer):
                network.append(
                    ResidualBlock(
                        in_channels=dim_out if j != 0 else dim_in,
                        out_channels=dim_out,
                        kernel_size=kernel_size,
                    )
                )

        # Define sequential network
        self.resnet = nn.Sequential(*network)

        # GRU network to be applied after the ResNet
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False
        )

    def forward(self, x):
        """Apply forward network.

        Args:
            x (tensor): input data.

        Returns:
            out: feature vector.
        """
        # Reshape input to the correct size
        x = x.view(-1, 1, 720)

        # Apply ResNet to input
        x = self.resnet(x)

        # Define zero hidden state for GRU
        h0 = torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim, device=x.device
        ).requires_grad_()

        # Apply GRU
        out, _ = self.gru(x, h0.detach())

        # Reshape output into the correct size
        out = out.reshape(-1, 256)

        return out

if __name__ == "__main__":
    model720 = ResnetGRU()

    from torchinfo import summary

    summary(model720, input_size=(1, 720), depth=5, verbose=1)