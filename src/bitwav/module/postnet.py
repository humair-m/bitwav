"""
PostNet module for refining mel spectrograms.
Adapted from: https://github.com/ming024/FastSpeech2
"""

import torch
import torch.nn as nn


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculates symmetric padding for 1D convolution."""
    return ((kernel_size - 1) * dilation) // 2


class Norm(nn.Module):
    """LayerNorm wrapper for 1D tensors (B, C, T)."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class PostNet(nn.Module):
    """
    Refines mel spectrograms using a stack of 1D convolutions with residual connection.
    """

    def __init__(
        self,
        input_channels: int = 100,
        channels: int = 512,
        kernel_size: int = 5,
        num_layers: int = 5,
        dropout: float = 0.5,
        use_layer_norm: bool = False,
    ):
        """
        Initializes PostNet.

        Args:
            input_channels (int): Input/output frequency bins.
            channels (int): Hidden channels.
            kernel_size (int): Convolution kernel size.
            num_layers (int): Number of convolutional layers.
            dropout (float): Dropout probability.
            use_layer_norm (bool): If True, uses LayerNorm instead of BatchNorm.
        """
        super().__init__()

        padding = get_padding(kernel_size)
        self.convolutions = nn.ModuleList()

        # Input layer
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(input_channels, channels, kernel_size=kernel_size, padding=padding),
                Norm(channels) if use_layer_norm else nn.BatchNorm1d(channels),
            )
        )
        
        # Intermediate layers
        for _ in range(1, num_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                    Norm(channels) if use_layer_norm else nn.BatchNorm1d(channels),
                )
            )
            
        # Output layer
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(channels, input_channels, kernel_size=kernel_size, padding=padding),
                Norm(input_channels) if use_layer_norm else nn.BatchNorm1d(input_channels),
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input mel spectrogram of shape (B, C, T).
            
        Returns:
            torch.Tensor: Refined mel spectrogram.
        """
        residual = x

        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)
            x = torch.tanh(x)
            x = self.dropout(x)

        x = self.convolutions[-1](x)
        x = self.dropout(x)

        return x + residual
