"""
Discriminator modules for GAN-based training of the Bitwav model.
Adapted from: https://github.com/gemelo-ai/vocos/blob/main/vocos/discriminators.py
"""

import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from typing import Tuple, List


def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    """Calculates asymmetric 2D padding for a given kernel size and dilation."""
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class SpectrogramDiscriminator(nn.Module):
    """
    Multi-band Spectrogram Discriminator.
    
    Splits the frequency axis into multiple bands and applies a 2D convolutional stack to each.
    """

    def __init__(
        self,
        frequency_bins: int,
        channels: int = 32,
        kernel_size: Tuple[int, int] = (3, 3),
        dilation: List[int] = [1, 2, 4],
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)),
        use_downsample: bool = True,
    ):
        """
        Initializes SpectrogramDiscriminator.

        Args:
            frequency_bins (int): Number of frequency bins in the input spectrogram.
            channels (int): Number of initial convolutional channels.
            kernel_size (Tuple[int, int]): Size of the 2D convolution kernel.
            dilation (List[int]): Dilations applied on the time axis.
            bands (Tuple[Tuple[float, float], ...]): Frequency bands as normalized ranges [0, 1].
            use_downsample (bool): Whether to apply downsampling at the end.
        """
        super().__init__()
        self.bands = [(int(b[0] * frequency_bins), int(b[1] * frequency_bins)) for b in bands]

        self.stacks = nn.ModuleList()
        for _ in self.bands:
            stack = nn.ModuleList(
                [weight_norm(nn.Conv2d(1, channels, kernel_size, padding=get_2d_padding(kernel_size)))]
            )

            for d in dilation:
                # Dilation applied only on the time axis
                pad = get_2d_padding(kernel_size, (d, 1))
                stack.append(weight_norm(nn.Conv2d(channels, channels, kernel_size, dilation=(d, 1), padding=pad)))

            stack.append(weight_norm(nn.Conv2d(channels, channels, kernel_size, padding=get_2d_padding(kernel_size))))
            self.stacks.append(stack)

        self.conv_post = weight_norm(nn.Conv2d(channels, 1, kernel_size, padding=get_2d_padding(kernel_size)))
        if use_downsample:
            self.downsample = nn.AvgPool2d(4, stride=2, padding=1, count_include_pad=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrogram of shape (B, C, F, T).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - score: Discriminator score map.
                - intermediates: List of intermediate feature maps for feature matching loss.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Rearrange to (B, C, T, F) for band splitting on last dimension
        x = rearrange(x, "b c f t -> b c t f")
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]

        results = []
        intermediates = []
        for x_band, stack in zip(x_bands, self.stacks):
            for layer in stack:
                x_band = layer(x_band)
                x_band = torch.nn.functional.leaky_relu(x_band, 0.1)
                intermediates.append(x_band)
            results.append(x_band)

        # Merge bands back together
        x = torch.cat(results, dim=-1)
        x = self.conv_post(x)
        x = self.downsample(x)
        return x, intermediates
