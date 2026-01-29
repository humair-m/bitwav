"""
Utility functions for Bitwav project, including audio loading, logging, and vocoding.
"""

import logging
from typing import Literal, List, Optional, Tuple

import torch
import torch.nn as nn

# Configure the Bitwav logger
logger = logging.getLogger("bitwav")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)


def get_logger() -> logging.Logger:
    """Returns the Bitwav logger instance."""
    return logger


def freeze_modules(modules: List[Optional[nn.Module]]):
    """
    Freezes the parameters of the given modules, disabling gradient computation.

    Args:
        modules (List[Optional[nn.Module]]): List of modules to freeze.
    """
    for module in modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False


def _load_audio_internal(
    path: str, frame_offset: Optional[int] = None, num_frames: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """
    Internal helper to load audio using soundfile.
    """
    import soundfile as sf

    with sf.SoundFile(path) as f:
        if frame_offset is not None:
            f.seek(frame_offset)
        frames = f.read(frames=num_frames or -1, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(frames.T)
        sample_rate = f.samplerate
    return waveform, sample_rate


def load_audio(audio_path: str, sample_rate: int = 24000) -> torch.Tensor:
    """
    Loads and preprocesses an audio file.
    
    Processing steps:
    1. Load using soundfile.
    2. Convert to mono.
    3. Resample to target sample rate.
    4. Normalize amplitude to range [-1, 1].

    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int): Desired output sample rate.

    Returns:
        torch.Tensor: Preprocessed mono waveform of shape (S,).
    """
    import torchaudio

    waveform, sr = _load_audio_internal(audio_path)

    # Convert to mono if it's multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to target rate if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Peak normalization
    max_val = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / max_val

    return waveform.squeeze(0)


def load_vocoder(name: Literal["vocos", "hift"] = "vocos") -> torch.nn.Module:
    """
    Loads a pretrained vocoder model.

    Args:
        name (Literal["vocos", "hift"]): Name of the vocoder to load.

    Returns:
        torch.nn.Module: The loaded vocoder model in eval mode.
    """
    if name == "vocos":
        from vocos import Vocos
        model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        return model.eval()
    
    elif name == "hift":
        from huggingface_hub import hf_hub_download
        from .module.hift import HiFTGenerator

        # Load weights from CosyVoice2
        model_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="hift.pt")
        model = HiFTGenerator()
        model.load_weights(model_path)
        return model.eval()
    
    else:
        raise ValueError(f"Unsupported vocoder name: {name}")


def vocode(vocoder: nn.Module, mel_spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes waveform from a mel spectrogram using the provided vocoder.

    Args:
        vocoder (nn.Module): Pretrained vocoder model.
        mel_spectrogram (torch.Tensor): Input mel spectrogram of shape (..., n_mels, T).

    Returns:
        torch.Tensor: Generated audio waveform.
    """
    mel_spectrogram = mel_spectrogram.to(torch.float32)
    vocoder_class_name = vocoder.__class__.__name__

    if "Vocos" in vocoder_class_name:
        generated_waveform = vocoder.decode(mel_spectrogram)
    elif "HiFT" in vocoder_class_name:
        generated_waveform = vocoder.inference(mel_spectrogram)
    else:
        raise ValueError(f"Unsupported vocoder class: {vocoder_class_name}")

    return generated_waveform
