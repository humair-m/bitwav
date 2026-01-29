"""
Adaptive Layer Normalization Zero (AdaLNZero) module.
Adapted from: https://github.com/facebookresearch/DiT
"""

import torch
from torch import nn
from typing import Tuple, Optional


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero-initialization for modulation and gating.
    
    This module predicts shift, scale, and optionally a gate value from a conditioning signal.
    Normalization is performed on the input features using non-learnable LayerNorm,
    and then modulated by the predicted shift and scale.
    """

    def __init__(
        self,
        dim: int,
        condition_dim: int,
        eps: float = 1e-5,
        return_gate: bool = True,
    ):
        """
        Initializes AdaLNZero.

        Args:
            dim (int): Feature dimension to be normalized.
            condition_dim (int): Dimension of the conditioning signal.
            eps (float): Epsilon for LayerNorm.
            return_gate (bool): Whether to predict and return a gating value.
        """
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.return_gate = return_gate

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        output_dim = 3 * dim if return_gate else 2 * dim
        self.condition_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, output_dim),
        )

        # Initialize the projection to zero so that at the start of training:
        # scale = 0, shift = 0, gate = 0.
        # This makes the block behave like an identity mapping initially.
        nn.init.zeros_(self.condition_proj[1].weight)
        nn.init.zeros_(self.condition_proj[1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input features of shape (B, L, dim).
            condition (torch.Tensor): Conditioning signal of shape (B, L, condition_dim) or (B, 1, condition_dim).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - modulated_x: Modulated features of shape (B, L, dim).
                - gate: Gating value if return_gate is True, else None.
        """
        x_norm = self.norm(x)
        condition_params = self.condition_proj(condition)

        if self.return_gate:
            shift, scale, gate = condition_params.chunk(3, dim=-1)
        else:
            shift, scale = condition_params.chunk(2, dim=-1)
            gate = None

        # Modulation: scale and shift
        # We add 1 to scale so that zero-initialized scale parameter results in identity scaling.
        modulated_x = x_norm * (1 + scale) + shift
        return modulated_x, gate
