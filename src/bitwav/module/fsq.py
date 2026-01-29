"""
Finite Scalar Quantization (FSQ) implementation based on:
https://arxiv.org/abs/2309.15505
"""

import torch
from torch import nn
from typing import List, Tuple, Dict

from ..util import get_logger

logger = get_logger()


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """
    Round with straight-through estimator gradients.
    
    This function performs rounding in the forward pass but copies the gradients
    from the output to the input in the backward pass.
    """
    zhat = z.round()
    return z + (zhat - z).detach()


def get_entropy(prob: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the entropy of a probability distribution.
    """
    return -torch.sum(prob * torch.log(prob + eps), dim=-1)


class FSQ(nn.Module):
    """
    Finite Scalar Quantizer core logic.
    
    Maps continuous vectors to a finite discrete set of points in a multidimensional space.
    """

    def __init__(self, levels: List[int]):
        """
        Initializes FSQ with the given levels for each dimension.

        Args:
            levels (List[int]): Number of levels for each dimension.
        """
        super().__init__()
        self.levels = levels
        self.dim = len(levels)

        _levels = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("_levels", _levels, persistent=False)
        
        # Basis for converting multi-dimensional indices to a single scalar index
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.long)
        self.register_buffer("_basis", _basis, persistent=False)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        Bound `z` to the range defined by levels.
        """
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantizes `z` and returns quantized `zhat` in normalized range [-1, 1].
        """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        """Scale and shift normalized quantized values to non-negative integer representation."""
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        """Inverse of _scale_and_shift."""
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """
        Converts quantized vectors to scalar indices.
        
        Args:
            zhat (torch.Tensor): Normalized quantized vectors of shape (..., D).
            
        Returns:
            torch.Tensor: Scalar indices of shape (...).
        """
        assert zhat.shape[-1] == len(self.levels)
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis.to(torch.float64)).to(torch.long).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts scalar indices back to normalized quantized vectors.
        """
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return self._scale_and_shift_inverse(codes_non_centered)

    def encode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes `z` into quantized vectors and indices.
        """
        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)
        return z_q, indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes indices back to quantized vectors.
        """
        return self.indices_to_codes(indices)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: returns quantized vectors and indices.
        """
        return self.encode(z)


class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantizer wrapper with projection layers.
    """

    def __init__(self, input_dim: int, output_dim: int, levels: List[int]) -> None:
        """
        Initializes FiniteScalarQuantizer.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            levels (List[int]): Number of levels for each quantization dimension.
        """
        super().__init__()
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim

        self.fsq = FSQ(levels)
        logger.debug(
            f"Finite Scalar Quantizer with levels: {levels}, input_dim: {input_dim}, output_dim: {output_dim}, codebook_size: {self.all_codebook_size}"
        )

        self.proj_in = nn.Linear(input_dim, len(levels)) if len(levels) != input_dim else nn.Identity()
        self.proj_out = nn.Linear(len(levels), output_dim) if len(levels) != output_dim else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self.output_dim_

    @property
    def all_codebook_size(self) -> int:
        """Returns the total number of possible discrete points in the codebook."""
        size = 1
        for level in self.fsq.levels:
            size *= level
        return size

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through the quantizer.

        Args:
            z (torch.Tensor): Input features of shape (B, T, C).

        Returns:
            z_q (torch.Tensor): Quantized and projected features.
            info_dict (Dict): Dictionary containing metadata like perplexity and indices.
        """
        latent = self.proj_in(z)
        quantized_latent, indices = self.fsq(latent)
        z_q = self.proj_out(quantized_latent)

        # Compute perplexity from used indices distribution
        flat_indices = indices.view(-1)
        unique_indices, counts = torch.unique(flat_indices, return_counts=True)
        used_indices_probs = counts.float() / flat_indices.numel()
        entropy = get_entropy(used_indices_probs)
        perplexity = torch.exp(entropy)

        info_dict = {
            "latent": latent,
            "quantized_latent": quantized_latent,
            "indices": indices,
            "perplexity": perplexity,
        }
        return z_q, info_dict

    def encode(self, z: torch.Tensor, skip_proj: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input features into indices.
        """
        z = self.proj_in(z)
        z_q, indices = self.fsq.encode(z)
        if not skip_proj:
            z_q = self.proj_out(z_q)
        return z_q, indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes indices back to features.
        """
        z_q = self.fsq.decode(indices)
        z_q = self.proj_out(z_q)
        return z_q
