"""
Audio dataset implementation with support for CSV metadata and windowed chunking.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from ..util import _load_audio_internal, get_logger

logger = get_logger()


@dataclass
class AudioItem:
    """Single item returned by the dataset."""
    waveform: torch.Tensor
    audio_id: str
    path: Path
    sample_rate: int
    frame_offset: Optional[int] = None


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Converts a multi-channel waveform to mono by averaging channels."""
    if waveform.shape[0] > 1:
        return torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def resample_audio(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resamples waveform to a new frequency."""
    if orig_freq != new_freq:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
        return resampler(waveform)
    return waveform


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Normalizes waveform to range [-1, 1]."""
    max_val = torch.max(torch.abs(waveform)) + 1e-8
    return waveform / max_val


def preprocess_audio(
    waveform: torch.Tensor, 
    sample_rate: int, 
    mono: bool, 
    normalize: bool, 
    target_sample_rate: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """Applies common preprocessing steps to raw audio."""
    if mono:
        waveform = convert_to_mono(waveform)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    if normalize:
        waveform = normalize_audio(waveform)
    return waveform, sample_rate


def pad_audio(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pads waveform with zeros along the temporal dimension to reach target_length."""
    current_length = waveform.shape[1]
    if current_length >= target_length:
        return waveform
    pad_length = target_length - current_length
    padding = torch.zeros((waveform.shape[0], pad_length), dtype=waveform.dtype, device=waveform.device)
    return torch.cat([waveform, padding], dim=1)


@dataclass
class ChunkInfo:
    """Stores metadata for a single audio chunk."""
    audio_id: str
    frame_offset: int  # In target sample rate
    num_frames: int  # In target sample rate


class ChunkedAudioDataset(Dataset):
    """
    Dataset that loads audio segments from files based on a CSV metadata file.
    Supports fixed-size chunking and resampling.
    """

    def __init__(
        self,
        csv_path: str,
        audio_root: str,
        chunk_size: Optional[int] = None,
        hop_size: Optional[int] = None,
        mono: bool = True,
        normalize: bool = True,
        target_sample_rate: Optional[int] = None,
    ):
        """
        Initializes ChunkedAudioDataset.

        Args:
            csv_path (str): Path to CSV with columns [audio_id, path, length, sample_rate].
            audio_root (str): Base path for audio files.
            chunk_size (Optional[int]): Fixed length of segments (in frames).
            hop_size (Optional[int]): Step between segments (defaults to chunk_size).
            mono (bool): Whether to force mono audio.
            normalize (bool): Whether to peak-normalize audio.
            target_sample_rate (Optional[int]): If set, audio will be resampled to this rate.
        """
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.chunk_size = chunk_size
        self.hop_size = hop_size if hop_size is not None else chunk_size
        self.mono = mono
        self.normalize = normalize
        self.target_sample_rate = target_sample_rate

        self.file_entries = self._load_csv()
        self.chunks = self._compute_chunks()
        logger.info(f"Dataset summary: {len(self.file_entries)} files, {len(self.chunks)} segments.")

    def _load_csv(self) -> Dict[str, Dict]:
        """Loads audio metadata from CSV file."""
        entries = {}
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries[row["audio_id"]] = {
                    "path": row["path"],
                    "length": int(row["length"]),
                    "sample_rate": int(row["sample_rate"]),
                }
        return entries

    def _compute_chunks(self) -> List[ChunkInfo]:
        """Calculates chunk boundaries for all files in the metadata."""
        chunks = []
        for audio_id, entry in self.file_entries.items():
            length = entry["length"]
            sample_rate = entry["sample_rate"]

            if self.target_sample_rate is not None and sample_rate != self.target_sample_rate:
                length = int(length * self.target_sample_rate / sample_rate)
                sample_rate = self.target_sample_rate

            if self.chunk_size is None or length <= self.chunk_size:
                chunks.append(ChunkInfo(audio_id=audio_id, frame_offset=0, num_frames=length))
            else:
                frame_offset = 0
                while frame_offset + self.chunk_size <= length:
                    chunks.append(ChunkInfo(audio_id=audio_id, frame_offset=frame_offset, num_frames=self.chunk_size))
                    frame_offset += self.hop_size

                # Ensure coverage of the tail end of the file
                last_start = length - self.chunk_size
                if last_start > frame_offset - self.hop_size:
                    chunks.append(ChunkInfo(audio_id=audio_id, frame_offset=last_start, num_frames=self.chunk_size))
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> AudioItem:
        """Loads and returns a single audio segment."""
        chunk = self.chunks[idx]
        entry = self.file_entries[chunk.audio_id]
        orig_sr = entry["sample_rate"]
        full_path = Path(self.audio_root) / entry["path"]

        # Map desired temporal boundaries back to original sample rate
        if self.target_sample_rate is not None and orig_sr != self.target_sample_rate:
            orig_offset = int(chunk.frame_offset * orig_sr / self.target_sample_rate)
            orig_num = int(chunk.num_frames * orig_sr / self.target_sample_rate)
        else:
            orig_offset, orig_num = chunk.frame_offset, chunk.num_frames

        waveform, sr = _load_audio_internal(full_path, frame_offset=orig_offset, num_frames=orig_num)
        waveform, sr = preprocess_audio(waveform, sr, self.mono, self.normalize, self.target_sample_rate)

        # Pad to chunk size if needed (e.g. truncated files)
        if self.chunk_size is not None and waveform.shape[1] < self.chunk_size:
            waveform = pad_audio(waveform, self.chunk_size)

        return AudioItem(
            waveform=waveform,
            audio_id=chunk.audio_id,
            path=full_path,
            sample_rate=sr,
            frame_offset=chunk.frame_offset
        )
