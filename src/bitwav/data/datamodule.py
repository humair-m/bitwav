"""
PyTorch Lightning DataModule for audio datasets.
Handles data loading, batching, and optional chunking.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from ..util import get_logger
from .dataset import AudioItem, ChunkedAudioDataset, pad_audio

logger = get_logger()


@dataclass
class AudioBatch:
    """
    Represents a batch of audio data.

    Attributes:
        waveform (torch.Tensor): Padded audio waveforms of shape (batch, channels, samples).
        audio_ids (List[str]): Identifiers for each audio item.
        paths (List[Path]): File system paths to the source audio files.
        sample_rates (List[int]): Sample rates for each audio item in the batch.
        frame_offsets (Optional[List[int]]): Offsets within the original files if chunked.
    """
    waveform: torch.Tensor
    audio_ids: List[str]
    paths: List[Path]
    sample_rates: List[int]
    frame_offsets: Optional[List[int]]


@dataclass
class AudioDataConfig:
    """
    Configuration for audio data loading.

    Attributes:
        csv_path (str): Path to the CSV containing audio metadata.
        audio_root (str): Root directory prepended to paths in the CSV.
        sample_rate (int): Target sample rate for resampling.
        mono (bool): Whether to convert audio to mono.
        normalize (bool): Whether to peak-normalize audio.
        chunk_size (Optional[int]): Fixed size for audio chunks in frames.
        chunk_hop_size (Optional[int]): Step size between chunks.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to pin memory in DataLoader.
        persistent_workers (bool): Whether to keep workers alive between epochs.
        shuffle (bool): Whether to shuffle the dataset.
        drop_last (bool): Whether to drop the last incomplete batch.
    """
    csv_path: str
    audio_root: str

    sample_rate: int = 24000
    mono: bool = True
    normalize: bool = True

    chunk_size: Optional[int] = None
    chunk_hop_size: Optional[int] = None

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = False
    drop_last: bool = False


def audio_collate_fn(batch: List[AudioItem]) -> AudioBatch:
    """
    Collates a list of AudioItems into an AudioBatch.
    Pads all waveforms to match the maximum length in the batch.
    """
    waveforms = [item.waveform for item in batch]
    max_length = max(wave.shape[1] for wave in waveforms)
    
    if any(wave.shape[1] != max_length for wave in waveforms):
        waveforms = [pad_audio(wave, max_length) for wave in waveforms]

    return AudioBatch(
        waveform=torch.stack(waveforms),
        audio_ids=[item.audio_id for item in batch],
        paths=[item.path for item in batch],
        sample_rates=[item.sample_rate for item in batch],
        frame_offsets=[item.frame_offset for item in batch],
    )


class AudioDataModule(L.LightningDataModule):
    """
    DataModule that manages training, validation, and test datasets for Bitwav.
    """

    def __init__(
        self,
        train_config: AudioDataConfig,
        val_config: Optional[AudioDataConfig] = None,
        test_config: Optional[AudioDataConfig] = None,
    ):
        """
        Initializes AudioDataModule.
        """
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config or train_config
        self.test_config = test_config or self.val_config

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _create_dataset(self, config: AudioDataConfig) -> Dataset:
        """Helper to instantiate ChunkedAudioDataset from config."""
        return ChunkedAudioDataset(
            csv_path=config.csv_path,
            audio_root=config.audio_root,
            chunk_size=config.chunk_size,
            hop_size=config.chunk_hop_size,
            mono=config.mono,
            normalize=config.normalize,
            target_sample_rate=config.sample_rate,
        )

    def setup(self, stage: Optional[str] = None):
        """Prepares datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(self.train_config)
            self.val_dataset = self._create_dataset(self.val_config)
        elif stage == "validate":
            self.val_dataset = self._create_dataset(self.val_config)
        elif stage == "test" or stage == "predict":
            self.test_dataset = self._create_dataset(self.test_config)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            persistent_workers=self.train_config.persistent_workers if self.train_config.num_workers > 0 else False,
            shuffle=self.train_config.shuffle,
            drop_last=self.train_config.drop_last,
            collate_fn=audio_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config.batch_size,
            num_workers=self.val_config.num_workers,
            pin_memory=self.val_config.pin_memory,
            persistent_workers=self.val_config.persistent_workers if self.val_config.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
            collate_fn=audio_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.test_config.num_workers,
            pin_memory=self.test_config.pin_memory,
            persistent_workers=self.test_config.persistent_workers if self.test_config.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
            collate_fn=audio_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
