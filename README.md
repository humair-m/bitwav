# Bitwav: A Disentangled Speech Tokenizer for Spoken Language Modeling

Bitwav is a speech tokenizer that encodes speech into compact content tokens and global embeddings and decodes them back to mel spectrograms. It features a custom 48kHz Vocos vocoder with dual-path synthesis for high-quality audio reconstruction.

## Features

- **Disentangled Representation**: Separates content (local tokens) from speaker characteristics (global embedding)
- **Finite Scalar Quantization (FSQ)**: Simple, stable discrete tokenization without codebook collapse
- **48kHz Vocoder**: Custom dual-path Vocos vocoder with Linkwitz-Riley crossover for high-fidelity audio
- **FlashAttention Support**: Efficient local window attention for fast training and inference

## Project Structure

```
bitwav/
├── src/bitwav/              # Main module
│   ├── model.py             # Core model architecture (BitwavModel)
│   ├── pipeline.py          # Training pipeline (LightningModule)
│   ├── data/                # Datasets and data loading
│   └── module/              # Neural network components
│       ├── transformer.py   # Transformer with RoPE and window attention
│       ├── fsq.py           # Finite Scalar Quantization
│       ├── global_encoder.py # Speaker embedding encoder
│       └── vocoder/         # 48kHz Vocos vocoder
├── config/                  # Configuration files
│   ├── model/               # Inference configs
│   └── train/               # Training configs
├── script/                  # Utility scripts
├── demo.ipynb               # Demo notebook
└── cli.py                   # Training CLI
```

## Installation

### For Inference

```bash
pip install git+https://github.com/humair-m/bitwav
```

### For Training

```bash
git clone https://github.com/humair-m/bitwav
cd bitwav
pip install -e ".[train]"
```

> [!IMPORTANT]
> We use [FlashAttention](https://github.com/Dao-AILab/flash-attention) for efficient local window attention. The model will fall back to PyTorch SDPA if FlashAttention is not available.

## Usage

### Basic Inference

```python
from bitwav import BitwavModel, load_audio, load_vocoder, vocode

# Load model
model = BitwavModel.from_pretrained("bitwav/bitwav-25hz-clean")
model = model.eval().cuda()

# Load vocoder
vocoder = load_vocoder(model.config.vocoder_name).cuda()

# Load and process audio
audio = load_audio("path/to/audio.wav", sample_rate=model.config.sample_rate).cuda()

# Encode: Extract content tokens and global embedding
features = model.encode(audio)
# features.content_token_indices: (seq_len,) - discrete content tokens
# features.global_embedding: (dim,) - speaker characteristics

# Decode: Reconstruct mel spectrogram
mel = model.decode(
    content_token_indices=features.content_token_indices,
    global_embedding=features.global_embedding,
)

# Vocode: Convert mel to waveform
waveform = vocode(vocoder, mel.unsqueeze(0))
```

## Training

1. **Prepare dataset metadata:**
```bash
python script/dump_dataset.py /path/to/LibriTTS --pattern "train-*/**/*.wav" -o data/libritts_train.csv
```

2. **Run training:**
```bash
python cli.py fit --config config/train/12hz_pretrain.yaml
```

## Available Models

| Model | Token Rate | Description |
|-------|------------|-------------|
| `bitwav-12.5hz` | 12.5 Hz | Lower token rate, more compression |
| `bitwav-25hz` | 25 Hz | Higher token rate, better quality |
| `bitwav-25hz-clean` | 25 Hz | Trained on clean LibriTTS-R data |

## License

MIT License
