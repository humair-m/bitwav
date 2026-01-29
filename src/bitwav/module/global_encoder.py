"""
Global encoder module for extracting speaker-level embeddings.
Based on ECAPA-TDNN's attentive statistics pooling.
"""

import torch
import torch.nn as nn
from typing import Optional

from .convnext import ConvNextBackbone


class AttentiveStatsPool(nn.Module):
    """
    Attentive Statistics Pooling as described in ECAPA-TDNN.
    Computes weighted mean and standard deviation along the time dimension.
    """

    def __init__(self, input_channels: int, output_channels: int, attention_channels: int = 128):
        """
        Initializes AttentiveStatsPool.

        Args:
            input_channels (int): Dimension of input features (channels).
            output_channels (int): Dimension of output embedding.
            attention_channels (int): Dimension of internal attention projection.
        """
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv1d(input_channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, input_channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.proj = nn.Linear(input_channels * 2, output_channels)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tension of shape (B, C, T).
            
        Returns:
            torch.Tensor: Global embedding of shape (B, output_channels).
        """
        alpha = self.attn(x)  # (B, C, T)

        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-4, max=1e4))

        x = torch.cat([mean, std], dim=1)  # (B, 2*C)
        return self.norm(self.proj(x))


class GlobalEncoder(nn.Module):
    """
    Global encoder that uses a ConvNext backbone followed by temporal pooling
    to extract a fixed-size global embedding from variable-length features.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        skip_embed: bool = False,
        attention_channels: int = 128,
        use_attn_pool: bool = True,
    ):
        """
        Initializes GlobalEncoder.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Dimension of the output global embedding.
            dim (int): Backbone internal dimension.
            intermediate_dim (int): Backbone intermediate dimension.
            num_layers (int): Number of backbone layers.
            skip_embed (bool): Whether to skip initial embedding projection.
            attention_channels (int): Channels for attentive pooling.
            use_attn_pool (bool): Whether to use attentive pooling or simple avg pooling.
        """
        super().__init__()

        self.backbone = ConvNextBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            skip_embed=skip_embed,
        )
        
        if use_attn_pool:
            self.pooling = AttentiveStatsPool(
                input_channels=dim, 
                output_channels=output_channels, 
                attention_channels=attention_channels
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
                nn.Linear(dim, output_channels),
                nn.LayerNorm(output_channels),
            )
        self.output_channels = output_channels

    @property
    def output_dim(self) -> int:
        return self.output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features of shape (B, T, C).
            
        Returns:
            torch.Tensor: Global embedding of shape (B, output_channels).
        """
        features = self.backbone(x)  # (B, T, C)
        
        # Pooling expects (B, C, T)
        features = features.transpose(1, 2)
        return self.pooling(features)
