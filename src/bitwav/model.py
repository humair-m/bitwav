import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import jsonargparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module.fsq import FiniteScalarQuantizer
from .module.global_encoder import GlobalEncoder
from .module.postnet import PostNet
from .module.ssl_extractor import SSLFeatureExtractor
from .module.transformer import Transformer
from .util import freeze_modules, get_logger

logger = get_logger()


@dataclass
class BitwavModelConfig:
    """
    Configuration for the Bitwav model.

    Attributes:
        local_ssl_layers (Tuple[int, ...]): Indices of SSL layers used for the local branch.
        global_ssl_layers (Tuple[int, ...]): Indices of SSL layers used for the global branch.
        normalize_ssl_features (bool): Whether to normalize local SSL features before encoding.
        downsample_factor (int): Temporal downsampling factor for local features.
        mel_upsample_factor (int): Conv1DTranspose upsampling factor for mel features before interpolation.
        use_conv_downsample (bool): Whether to use Conv1D for downsampling instead of average pooling.
        local_interpolation_mode (str): Interpolation mode for local upsampling ("linear", "nearest").
        mel_interpolation_mode (str): Interpolation mode for mel upsampling ("linear", "nearest").
        sample_rate (int): Audio sample rate.
        n_fft (int): FFT size for mel spectrogram calculation.
        hop_length (int): Hop length for mel spectrogram calculation.
        n_mels (int): Number of mel bins.
        padding (str): Padding mode for mel spectrogram ("center", "same", etc.).
        mel_fmin (int): Minimum frequency for mel spectrograms.
        mel_fmax (int | None): Maximum frequency for mel spectrograms.
        bigvgan_style_mel (bool): Whether to use BigVGAN-style mel spectrograms.
        vocoder_name (Literal["vocos", "hift"]): Vocoder to use for waveform synthesis.
    """
    local_ssl_layers: Tuple[int, ...] = (6, 9)
    global_ssl_layers: Tuple[int, ...] = (1, 2)
    normalize_ssl_features: bool = True

    downsample_factor: int = 2
    mel_upsample_factor: int = 4
    use_conv_downsample: bool = True
    local_interpolation_mode: str = "linear"
    mel_interpolation_mode: str = "linear"

    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"
    mel_fmin: int = 0
    mel_fmax: Optional[int] = None
    bigvgan_style_mel: bool = False

    vocoder_name: Literal["vocos", "hift"] = "vocos"


@dataclass
class BitwavFeatures:
    """
    Features extracted by the Bitwav model.

    Attributes:
        content_embedding (torch.Tensor | None): Content embeddings of shape (seq_len, dim).
        content_token_indices (torch.Tensor | None): Content token indices of shape (seq_len,).
        global_embedding (torch.Tensor | None): Global embedding of shape (dim,).
    """
    content_embedding: Optional[torch.Tensor] = None
    content_token_indices: Optional[torch.Tensor] = None
    global_embedding: Optional[torch.Tensor] = None


class BitwavModel(nn.Module):
    """
    Bitwav model architecture for spoken language modeling.
    
    This model encodes speech into compact content tokens and global embeddings,
    and decodes them back to mel spectrograms.
    """

    def __init__(
        self,
        config: BitwavModelConfig,
        ssl_feature_extractor: SSLFeatureExtractor,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        global_encoder: GlobalEncoder,
        mel_prenet: Transformer,
        mel_decoder: Transformer,
        mel_postnet: PostNet,
        feature_decoder: Optional[Transformer] = None,
    ):
        """
        Initializes the BitwavModel.

        Args:
            config (BitwavModelConfig): Configuration object.
            ssl_feature_extractor (SSLFeatureExtractor): SSL feature extractor module.
            local_encoder (Transformer): Transformer encoder for local branch.
            local_quantizer (FiniteScalarQuantizer): Quantizer for local features.
            global_encoder (GlobalEncoder): Encoder for global branch.
            mel_prenet (Transformer): Pre-net for mel decoder.
            mel_decoder (Transformer): Decoder for generating mel spectrograms.
            mel_postnet (PostNet): Post-net for mel spectrograms.
            feature_decoder (Optional[Transformer]): Optional decoder for SSL feature reconstruction.
        """
        super().__init__()
        self.config = config
        self._init_ssl_extractor(config, ssl_feature_extractor)
        self._init_local_branch(config, local_encoder, local_quantizer, feature_decoder)
        self._init_global_branch(global_encoder)
        self._init_mel_decoder(config, mel_prenet, mel_decoder, mel_postnet)

    def _init_ssl_extractor(self, config: BitwavModelConfig, ssl_feature_extractor: SSLFeatureExtractor):
        """Initialize and configure SSL feature extractor."""
        self.ssl_feature_extractor = ssl_feature_extractor
        freeze_modules([self.ssl_feature_extractor])
        logger.debug(
            f"SSL feature extractor initialized and frozen, feature dim: {self.ssl_feature_extractor.feature_dim}"
        )

        self.local_ssl_layers = list(config.local_ssl_layers)
        self.global_ssl_layers = list(config.global_ssl_layers)

        if config.normalize_ssl_features:
            logger.debug("Normalizing local SSL features before encoding")

    def _init_local_branch(
        self,
        config: BitwavModelConfig,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Optional[Transformer],
    ):
        """Initialize local branch components (encoder, downsampling, quantizer, decoder)."""
        self.local_encoder = local_encoder
        self.local_quantizer = local_quantizer
        self.feature_decoder = feature_decoder

        self.downsample_factor = config.downsample_factor
        if self.downsample_factor > 1:
            if config.use_conv_downsample:
                feature_dim = local_encoder.output_dim
                self.conv_downsample = nn.Conv1d(
                    feature_dim, feature_dim, kernel_size=config.downsample_factor, stride=config.downsample_factor
                )
                self.conv_upsample = nn.ConvTranspose1d(
                    feature_dim, feature_dim, kernel_size=config.downsample_factor, stride=config.downsample_factor
                )
            else:
                self.conv_downsample = None
                self.conv_upsample = None
        else:
            self.conv_downsample = None
            self.conv_upsample = None

    def _init_global_branch(self, global_encoder: GlobalEncoder):
        """Initialize global branch components."""
        self.global_encoder = global_encoder

    def _init_mel_decoder(
        self, config: BitwavModelConfig, mel_prenet: Transformer, mel_decoder: Transformer, mel_postnet: PostNet
    ):
        """Initialize mel decoder components (prenet, upsampling, decoder, postnet)."""
        self.mel_prenet = mel_prenet
        self.mel_decoder = mel_decoder
        self.mel_postnet = mel_postnet

        self.mel_conv_upsample = None
        if config.mel_upsample_factor > 1:
            input_dim = mel_prenet.output_dim
            self.mel_conv_upsample = nn.ConvTranspose1d(
                input_dim, input_dim, kernel_size=config.mel_upsample_factor, stride=config.mel_upsample_factor
            )

    def _calculate_waveform_padding(self, audio_length: int, ensure_recon_length: bool = False) -> int:
        """Calculate required padding for input waveform to ensure consistent SSL feature lengths."""
        extractor = self.ssl_feature_extractor
        sample_rate = self.config.sample_rate
        num_samples_after_resampling = audio_length / sample_rate * extractor.ssl_sample_rate
        expected_ssl_output_length = math.ceil(num_samples_after_resampling / extractor.hop_size)
        
        if ensure_recon_length and (remainder := expected_ssl_output_length % self.downsample_factor) != 0:
            expected_ssl_output_length += self.downsample_factor - remainder
            
        num_samples_required_after_resampling = extractor.get_minimum_input_length(expected_ssl_output_length)
        num_samples_required = num_samples_required_after_resampling / extractor.ssl_sample_rate * sample_rate
        padding = math.ceil((num_samples_required - audio_length) / 2)
        return padding

    def _calculate_original_audio_length(self, token_length: int) -> int:
        """Calculate the original audio length based on token length."""
        extractor = self.ssl_feature_extractor
        sample_rate = self.config.sample_rate
        feature_length = token_length * self.downsample_factor
        num_samples_required_after_resampling = extractor.get_minimum_input_length(feature_length)
        num_samples_required = num_samples_required_after_resampling / extractor.ssl_sample_rate * sample_rate
        return math.ceil(num_samples_required)

    def _calculate_target_mel_length(self, audio_length: int) -> int:
        """Calculate the target mel spectrogram length based on audio length."""
        if self.config.padding == "center":
            return audio_length // self.config.hop_length + 1
        elif self.config.padding == "same":
            return audio_length // self.config.hop_length
        else:
            return (audio_length - self.config.n_fft) // self.config.hop_length + 1

    def _process_ssl_features(self, features: list[torch.Tensor], layers: list[int]) -> torch.Tensor:
        """Process SSL features by averaging selected layers."""
        if len(layers) > 1:
            selected_features = [features[i - 1] for i in layers]
            mixed_features = torch.stack(selected_features, dim=0).mean(dim=0)
        else:
            mixed_features = features[layers[0] - 1]
        return mixed_features

    def _normalize_ssl_features(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize SSL features across time steps."""
        if not self.config.normalize_ssl_features:
            return features

        mean = torch.mean(features, dim=1, keepdim=True)
        std = torch.std(features, dim=1, keepdim=True)
        return (features - mean) / (std + eps)

    def forward_ssl_features(
        self, waveform: torch.Tensor, padding: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to extract SSL features.

        Args:
            waveform: Input waveform tensor of shape (B, channels, samples).
            padding: Optional padding to apply on both sides of the waveform.

        Returns:
            local_ssl_features: Local SSL features. (B, T, C)
            global_ssl_features: Global SSL features. (B, T, C)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        if padding is not None and padding > 0:
            waveform = F.pad(waveform, (padding, padding), mode="constant")

        with torch.no_grad():
            ssl_features = self.ssl_feature_extractor(waveform)

        local_ssl_features = self._process_ssl_features(ssl_features, self.local_ssl_layers)
        local_ssl_features = self._normalize_ssl_features(local_ssl_features)

        global_ssl_features = self._process_ssl_features(ssl_features, self.global_ssl_layers)

        return local_ssl_features, global_ssl_features

    def forward_content(
        self, local_ssl_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass to extract content embeddings from the local branch.

        Args:
            local_ssl_features: Local SSL features tensor of shape (B, T, C).

        Returns:
            local_quantized: Quantized local embeddings. (B, T/factor, C)
            indices: Content token indices. (B, T/factor)
            ssl_recon: Reconstructed SSL features (if feature decoder is present). (B, T, C)
            perplexity: Quantizer perplexity.
        """
        local_encoded = self.local_encoder(local_ssl_features)

        if self.downsample_factor > 1:
            if self.config.use_conv_downsample:
                local_encoded = self.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)
            else:
                local_encoded = F.avg_pool1d(
                    local_encoded.transpose(1, 2), kernel_size=self.downsample_factor, stride=self.downsample_factor
                ).transpose(1, 2)

        ssl_recon = None
        perplexity = torch.tensor(0.0)
        if self.feature_decoder is not None:
            local_quantized, local_quantize_info = self.local_quantizer(local_encoded)
            indices = local_quantize_info["indices"]
            perplexity = torch.mean(local_quantize_info["perplexity"])

            local_latent_for_ssl = local_quantized
            if self.downsample_factor > 1:
                if self.config.use_conv_downsample:
                    local_latent_for_ssl = self.conv_upsample(local_latent_for_ssl.transpose(1, 2)).transpose(1, 2)
                else:
                    local_latent_for_ssl = F.interpolate(
                        local_latent_for_ssl.transpose(1, 2),
                        size=local_ssl_features.shape[1],
                        mode=self.config.local_interpolation_mode,
                    ).transpose(1, 2)

            ssl_recon = self.feature_decoder(local_latent_for_ssl)
        else:
            local_quantized, indices = self.local_quantizer.encode(local_encoded)

        return local_quantized, indices, ssl_recon, perplexity

    def forward_global(self, global_ssl_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract global embeddings from the global branch.

        Args:
            global_ssl_features: Global SSL features tensor of shape (B, T, C).

        Returns:
            global_encoded: Global embeddings. (B, C)
        """
        global_encoded = self.global_encoder(global_ssl_features)
        return global_encoded

    def forward_mel(
        self, content_embeddings: torch.Tensor, global_embeddings: torch.Tensor, mel_length: int
    ) -> torch.Tensor:
        """
        Forward pass to generate mel spectrogram from content and global embeddings.

        Args:
            content_embeddings: Content embeddings tensor of shape (B, T, C).
            global_embeddings: Global embeddings tensor of shape (B, C).
            mel_length: Target mel spectrogram length.

        Returns:
            mel_recon: Reconstructed mel spectrogram. (B, n_mels, T_mel)
        """
        local_latent = self.mel_prenet(content_embeddings)

        if self.mel_conv_upsample is not None:
            local_latent = self.mel_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
        local_latent = F.interpolate(
            local_latent.transpose(1, 2), size=mel_length, mode=self.config.mel_interpolation_mode
        ).transpose(1, 2)

        mel_recon = self.mel_decoder(local_latent, condition=global_embeddings.unsqueeze(1))
        mel_recon = mel_recon.transpose(1, 2)

        mel_recon = self.mel_postnet(mel_recon)
        return mel_recon

    def weights_to_save(self, *, include_modules: list[str]) -> dict[str, torch.Tensor]:
        """Get model weights for saving. Excludes modules not needed for inference."""
        excluded_modules = [
            m for m in ["ssl_feature_extractor", "feature_decoder", "conv_upsample"] if m not in include_modules
        ]
        state_dict = {
            name: param
            for name, param in self.named_parameters()
            if not any(name.startswith(excl) for excl in excluded_modules)
        }
        return state_dict

    @classmethod
    def from_hparams(cls, config_path: str) -> "BitwavModel":
        """Instantiate BitwavModel from a config file."""
        parser = jsonargparse.ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=BitwavModel)
        cfg = parser.parse_path(config_path)
        cfg = parser.instantiate_classes(cfg)
        return cfg.model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
    ) -> "BitwavModel":
        """Load BitwavModel from HuggingFace Hub or local files."""
        if repo_id is not None:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id, "config.yaml", revision=revision)
            weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
        else:
            if config_path is None or weights_path is None:
                raise ValueError("Provide either repo_id or both config_path and weights_path.")

        model = cls.from_hparams(config_path)
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from: {weights_path}")
        return model

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, return_content: bool = True, return_global: bool = True) -> BitwavFeatures:
        """Extract content and/or global features from audio."""
        audio_length = waveform.size(0)
        padding = self._calculate_waveform_padding(audio_length)
        local_ssl_features, global_ssl_features = self.forward_ssl_features(waveform.unsqueeze(0), padding=padding)

        result = BitwavFeatures()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            if return_content:
                content_embedding, token_indices, _, _ = self.forward_content(local_ssl_features)
                result.content_embedding = content_embedding.squeeze(0)
                result.content_token_indices = token_indices.squeeze(0)

            if return_global:
                global_embedding = self.forward_global(global_ssl_features)
                result.global_embedding = global_embedding.squeeze(0)

        return result

    def decode_token_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode content token indices to embeddings."""
        return self.local_quantizer.decode(indices)

    @torch.inference_mode()
    def decode(
        self,
        global_embedding: torch.Tensor,
        content_token_indices: Optional[torch.Tensor] = None,
        content_embedding: Optional[torch.Tensor] = None,
        target_audio_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Synthesize audio mel spectrogram from features."""
        if content_embedding is None:
            if content_token_indices is None:
                raise ValueError("Either content_token_indices or content_embedding must be provided.")
            content_embedding = self.decode_token_indices(content_token_indices)

        if target_audio_length is None:
            seq_len = content_embedding.size(0)
            target_audio_length = self._calculate_original_audio_length(seq_len)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            mel_length = self._calculate_target_mel_length(target_audio_length)
            content_embedding = content_embedding.unsqueeze(0)
            global_embedding = global_embedding.unsqueeze(0)
            mel_spectrogram = self.forward_mel(content_embedding, global_embedding, mel_length=mel_length)

        return mel_spectrogram.squeeze(0)

    @torch.inference_mode()
    def voice_conversion(self, source_waveform: torch.Tensor, reference_waveform: torch.Tensor) -> torch.Tensor:
        """Perform voice conversion."""
        source_features = self.encode(source_waveform, return_content=True, return_global=False)
        reference_features = self.encode(reference_waveform, return_content=False, return_global=True)

        return self.decode(
            content_embedding=source_features.content_embedding,
            global_embedding=reference_features.global_embedding,
            target_audio_length=source_waveform.size(0),
        )
