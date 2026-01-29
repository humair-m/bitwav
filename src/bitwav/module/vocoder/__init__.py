"""
Bitwav 48kHz Vocoder Module.

This module provides a high-quality 48kHz neural vocoder based on Vocos,
using a dual-path architecture with Linkwitz-Riley crossover merging for
superior audio quality.

Components:
    - Vocos: Main vocoder class with 48kHz output capability
    - UpSamplerBlock: Upsampling block for 48kHz synthesis
    - crossover_merge_linkwitz_riley: Crossover merging function

Example:
    >>> from bitwav.module.vocoder import Vocos
    >>> vocoder = Vocos.from_pretrained("repo_id")
    >>> audio = vocoder.decode(features)
"""

from .vocos import Vocos, instantiate_class
from .upsampler_block import UpSamplerBlock
from .linkwitz import crossover_merge_linkwitz_riley

__all__ = [
    "Vocos",
    "UpSamplerBlock",
    "crossover_merge_linkwitz_riley",
    "instantiate_class",
]
