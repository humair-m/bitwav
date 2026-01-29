"""
Audio feature extraction modules for mel spectrograms.
Supports Vocos-style and BigVGAN-style mel spectrograms.
"""

import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from typing import Optional


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """Computes log of x with a minimum clip value to avoid -inf."""
    return torch.log(torch.clip(x, min=clip_val))


class MelSpectrogramFeature(nn.Module):
    """
    Module for extracting mel spectrograms from audio waveforms.
    
    Supports two styles:
    1. Vocos style: Uses torchaudio MelSpectrogram (HTK mel scale, linear scaling).
    2. BigVGAN style: Uses librosa mel base (Slaney mel scale, area normalization).
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        padding: str = "center",
        fmin: int = 0,
        fmax: Optional[int] = None,
        bigvgan_style_mel: bool = False,
    ):
        """
        Initializes MelSpectrogramFeature.

        Args:
            sample_rate (int): Audio sample rate.
            n_fft (int): FFT size.
            hop_length (int): Hop size.
            n_mels (int): Number of mel bins.
            padding (str): Padding mode ('center' or 'same').
            fmin (int): Minimum frequency.
            fmax (Optional[int]): Maximum frequency.
            bigvgan_style_mel (bool): If True, uses BigVGAN/HiFT style calculation.
        """
        super().__init__()

        self.bigvgan_style_mel = bigvgan_style_mel
        if bigvgan_style_mel:
            self.n_fft = n_fft
            self.win_size = n_fft
            self.hop_size = hop_length
            
            mel_basis = librosa_mel_fn(
                sr=sample_rate, n_fft=n_fft, n_mels=n_mels, norm="slaney", htk=False, fmin=fmin, fmax=fmax
            )
            mel_basis = torch.from_numpy(mel_basis).float()
            hann_window = torch.hann_window(n_fft)
            self.register_buffer("mel_basis", mel_basis)
            self.register_buffer("hann_window", hann_window)
        else:
            if padding not in ["center", "same"]:
                raise ValueError("Padding must be 'center' or 'same'.")

            self.padding = padding
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                center=padding == "center",
                power=1,
                fmin=fmin,
                fmax=fmax,
            )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extracts mel spectrogram from audio.

        Args:
            audio (torch.Tensor): Audio waveform of shape (B, S).

        Returns:
            torch.Tensor: Mel spectrogram of shape (B, n_mels, T).
        """
        if self.bigvgan_style_mel:
            return self.bigvgan_mel(audio)
        else:
            return self.vocos_mel(audio)

    def vocos_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Extracts mel spectrogram using Vocos-style configuration."""
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")

        specgram = self.mel_spec.spectrogram(audio)
        mel_specgram = self.mel_spec.mel_scale(specgram)
        return safe_log(mel_specgram)

    def bigvgan_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Extracts mel spectrogram using BigVGAN-style configuration."""
        padding = (self.n_fft - self.hop_size) // 2
        audio = torch.nn.functional.pad(audio, (padding, padding), mode="reflect")
        audio = audio.reshape(-1, audio.shape[-1])

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.reshape(audio.shape[:-1] + spec.shape[-2:])

        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(self.mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec
