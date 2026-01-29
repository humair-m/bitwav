"""
ConvNeXt blocks and backbone adapted for 1D audio signals.
Adapted from: https://github.com/gemelo-ai/vocos/blob/main/vocos/models.py
"""

import torch
from torch import nn
from typing import Optional


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block for 1D sequences.

    Architecture:
    1. Depthwise Convolution (kernel size 7)
    2. LayerNorm (on channel dimension)
    3. Pointwise Convolution (Linear layer)
    4. GELU Activation
    5. Pointwise Convolution (Linear layer)
    6. Layer Scale (optional)
    7. Residual Connection
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ):
        """
        Initializes ConvNeXtBlock.

        Args:
            dim (int): Input/output dimension.
            intermediate_dim (int): Intermediate dimension for the pointwise expansion.
            layer_scale_init_value (float): Initial value for the layer scale parameter.
        """
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
        """
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNextBackbone(nn.Module):
    """
    Backbone module composed of multiple ConvNeXt blocks.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        output_channels: Optional[int] = None,
        layer_scale_init_value: Optional[float] = None,
        skip_embed: bool = False,
    ):
        """
        Initializes ConvNextBackbone.

        Args:
            input_channels (int): Number of channels in the input signal.
            dim (int): Hidden dimension of the blocks.
            intermediate_dim (int): Intermediate dimension for ConvNeXtBlocks.
            num_layers (int): Number of ConvNeXtBlocks.
            output_channels (Optional[int]): Optional projection to a different output dimension.
            layer_scale_init_value (Optional[float]): Initial value for layer scale.
            skip_embed (bool): If True, skips the initial embedding convolution.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dim = dim
        
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3) if not skip_embed else nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        ls_init = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=ls_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(dim, output_channels) if output_channels else nn.Identity()
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    @property
    def input_dim(self) -> int:
        return self.input_channels

    @property
    def output_dim(self) -> int:
        return self.output_channels if self.output_channels else self.dim

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, H).
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.proj_out(x)
        return x
