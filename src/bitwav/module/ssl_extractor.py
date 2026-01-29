"""
SSL feature extractor module using torchaudio's Wav2Vec2/HuBERT/WavLM models.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.pipelines as pipelines
from torchaudio.models.wav2vec2 import Wav2Vec2Model
from torchaudio.models.wav2vec2.components import ConvLayerBlock
from typing import List, Tuple, Optional, Union

from ..util import get_logger

logger = get_logger()

# Map of friendly names to torchaudio pipeline bundles
MODEL_REGISTRY = {
    "wav2vec2_base": pipelines.WAV2VEC2_BASE,
    "wav2vec2_large": pipelines.WAV2VEC2_LARGE,
    "wav2vec2_large_lv60k": pipelines.WAV2VEC2_LARGE_LV60K,
    "hubert_base": pipelines.HUBERT_BASE,
    "hubert_large": pipelines.HUBERT_LARGE,
    "hubert_xlarge": pipelines.HUBERT_XLARGE,
    "wavlm_base": pipelines.WAVLM_BASE,
    "wavlm_base_plus": pipelines.WAVLM_BASE_PLUS,
    "wavlm_large": pipelines.WAVLM_LARGE,
}


class SSLFeatureExtractor(nn.Module):
    """
    Module for extracting hidden states from SSL models (WavLM, HuBERT, Wav2Vec2).
    """

    def __init__(self, model_name: str = "wavlm_base_plus", output_layer: Optional[int] = None, sample_rate: int = 16000):
        """
        Initializes SSLFeatureExtractor.

        Args:
            model_name (str): The name of the SSL model to use.
            output_layer (Optional[int]): Which layer's features to extract (None for all until the default).
            sample_rate (int): The sample rate of the input audio to this module.
        """
        super().__init__()
        self.output_layer = output_layer if output_layer is not None else -1

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
        
        bundle = MODEL_REGISTRY[model_name]
        self.model: Wav2Vec2Model = bundle.get_model()
        self.model.eval()
        self.feature_dim: int = bundle._params["encoder_embed_dim"]
        self.ssl_sample_rate = bundle.sample_rate

        # Initialize resampler if input audio SR differs from model SR
        if sample_rate != self.ssl_sample_rate:
            logger.debug(f"Resampling from {sample_rate} to {self.ssl_sample_rate} required by {model_name}.")
            self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.ssl_sample_rate)
        else:
            self.resampler = None

    @property
    def hop_size(self) -> int:
        """Determines the effective temporal hop size of the feature extractor's CNN backbone."""
        hop_size = 1
        for _, stride in self.conv_config:
            hop_size *= stride
        return hop_size

    @property
    def conv_config(self) -> List[Tuple[int, int]]:
        """Extracts the kernel size and stride configuratons of the CNN encoder."""
        conv_layers = []
        for layer in self.model.feature_extractor.conv_layers:
            layer: ConvLayerBlock
            conv_layers.append((layer.kernel_size, layer.stride))
        return conv_layers

    def get_minimum_input_length(self, desired_output_length: int) -> int:
        """
        Calculates the required input length (samples) to produce a specific number of feature frames.
        Uses the inverse CNN transformation logic.
        """
        length = desired_output_length
        for kernel_size, stride in reversed(self.conv_config):
            length = (length - 1) * stride + kernel_size
        return length

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        num_layers: Optional[int] = None,
        return_lengths: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        Extracts multi-layer hidden states from the SSL model.

        Args:
            waveform (torch.Tensor): Audio waveform of shape (B, S).
            lengths (Optional[torch.Tensor]): Waveform lengths for padding handling.
            num_layers (Optional[int]): Number of layers to extract.
            return_lengths (bool): Whether to return resulting feature lengths.

        Returns:
            List[torch.Tensor]: A list of feature tensors, one for each layer.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        if self.resampler is not None:
            waveform = self.resampler(waveform)

        features, feature_lengths = self.model.extract_features(
            waveform, lengths, num_layers=num_layers or self.output_layer
        )

        if return_lengths:
            return features, feature_lengths
        return features
