"""
HiFT vocoder generator implementation.
Combines a Neural Source Filter with ISTFTNet.
Adapted from: 
- https://github.com/yl4579/HiFTNet/blob/main/models.py
- https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/hifigan/generator.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from torch.distributions.uniform import Uniform
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Dict, List, Optional, Tuple


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Snake(nn.Module):
    """
    Snake activation function for periodic signals.
    Snake(x) := x + 1/alpha * sin^2(x * alpha)
    """

    def __init__(self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class ResBlock(nn.Module):
    """Residual block with snake activation."""

    def __init__(self, channels: int = 512, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                                                   dilation=dilation, padding=get_padding(kernel_size, dilation))))
            self.convs2.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                                                   dilation=1, padding=get_padding(kernel_size, 1))))
        
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        
        self.activations1 = nn.ModuleList([Snake(channels) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])


class ConvRNNF0Predictor(nn.Module):
    """Predicts F0 from mel spectrograms."""
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(cond_channels, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class SineGen(nn.Module):
    """Harmonic sine generator for NSF."""

    def __init__(self, samp_rate: int, upsample_scale: int, harmonic_num: int = 0, 
                 sine_amp: float = 0.1, noise_std: float = 0.003, voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def forward(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        rad_values = (fn / self.sampling_rate) % 1
        
        # Initial phase noise
        rand_ini = torch.rand(fn.shape[0], fn.shape[2], device=fn.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini

        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
        sine_waves = torch.sin(phase) * self.sine_amp

        uv = (f0 > self.voiced_threshold).float()
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        
        return sine_waves * uv + noise, uv, noise


class SourceModuleHnNSF(nn.Module):
    """Neural Source filter excitation generator."""

    def __init__(self, sampling_rate: int, upsample_scale: int, harmonic_num: int = 0, 
                 sine_amp: float = 0.1, add_noise_std: float = 0.003, voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv

    def remove_weight_norm(self):
        pass


class HiFTGenerator(nn.Module):
    """
    Full HiFTNet generator (ISTFTNet with source filtering).
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 5, 3],
        upsample_kernel_sizes: List[int] = [16, 11, 7],
        istft_n_fft: int = 16,
        istft_hop_len: int = 4,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor_channels: int = 512,
    ):
        super().__init__()
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_n_fft = istft_n_fft
        self.istft_hop_len = istft_hop_len
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        full_scale = np.prod(upsample_rates) * istft_hop_len
        self.m_source = SourceModuleHnNSF(sampling_rate, full_scale, nb_harmonics, 
                                          nsf_alpha, nsf_sigma, nsf_voiced_threshold)
        self.f0_upsamp = nn.Upsample(scale_factor=full_scale)

        self.conv_pre = weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(base_channels // (2**i), 
                                                       base_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        down_cum = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(down_cum[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(Conv1d(istft_n_fft + 2, base_channels // (2 ** (i + 1)), 1, 1))
            else:
                self.source_downs.append(Conv1d(istft_n_fft + 2, base_channels // (2 ** (i + 1)), u * 2, u, padding=(u // 2)))
            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, istft_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.register_buffer("stft_window", torch.from_numpy(get_window("hann", istft_n_fft, fftbins=True).astype(np.float32)))
        self.f0_predictor = ConvRNNF0Predictor(in_channels=in_channels, cond_channels=f0_predictor_channels)

    def remove_weight_norm(self):
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        self.m_source.remove_weight_norm()
        for l in self.source_downs: remove_weight_norm(l)
        for l in self.source_resblocks: l.remove_weight_norm()

    def _stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(x, self.istft_n_fft, self.istft_hop_len, self.istft_n_fft, 
                          window=self.stft_window, return_complex=True)
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        return torch.istft(torch.complex(real, img), self.istft_n_fft, self.istft_hop_len, 
                           self.istft_n_fft, window=self.stft_window)

    def decode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_real, s_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            
            si = self.source_resblocks[i](self.source_downs[i](s_stft))
            x = x + si
            
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.conv_post(F.leaky_relu(x))
        mag = torch.exp(x[:, : self.istft_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_n_fft // 2 + 1 :, :])
        
        wav = self._istft(mag, phase)
        return torch.clamp(wav, -self.audio_limit, self.audio_limit)

    def forward(self, speech_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = speech_feat.transpose(1, 2)
        f0 = self.f0_predictor(x)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s_merge, _, _ = self.m_source(s)
        return self.decode(x, s_merge.transpose(1, 2)), f0

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor) -> torch.Tensor:
        wav, _ = self.forward(speech_feat)
        return wav
