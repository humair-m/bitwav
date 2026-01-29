import os
import tarfile
import urllib.request
from pathlib import Path
import csv
import soundfile as sf
from tqdm import tqdm

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)

def setup_ljspeech(output_dir="data/ljspeech"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = output_path / "LJSpeech-1.1.tar.bz2"
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    
    if not (output_path / "LJSpeech-1.1").exists():
        if not tar_path.exists():
            download_file(url, tar_path)
        
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=output_path)
    else:
        print("LJSpeech-1.1 already extracted.")

    wavs_dir = output_path / "LJSpeech-1.1" / "wavs"
    metadata_path = output_path / "metadata.csv"
    
    print("Generating metadata.csv...")
    wav_files = list(wavs_dir.glob("*.wav"))
    
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_id", "path", "length", "sample_rate"])
        writer.writeheader()
        
        for wav_path in tqdm(wav_files, desc="Processing wavs"):
            info = sf.info(str(wav_path))
            writer.writerow({
                "audio_id": wav_path.stem,
                "path": str(wav_path.relative_to(output_path)),
                "length": info.frames,
                "sample_rate": info.samplerate
            })
            
    print(f"Dataset setup complete. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    setup_ljspeech()
