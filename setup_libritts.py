import os
import tarfile
import urllib.request
from pathlib import Path
import csv
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

SUBSETS = {
    "train-clean-100": "http://www.openslr.org/resources/60/train-clean-100.tar.gz",
    "dev-clean": "http://www.openslr.org/resources/60/dev-clean.tar.gz",
    "test-clean": "http://www.openslr.org/resources/60/test-clean.tar.gz",
}

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)

def process_wav(wav_path, output_path):
    try:
        info = torchaudio.info(str(wav_path))
        return {
            "audio_id": wav_path.stem,
            "path": str(wav_path.relative_to(output_path)),
            "length": info.num_frames,
            "sample_rate": info.sample_rate
        }
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def setup_libritts(output_dir="data/libritts", subset_names=None):
    if subset_names is None:
        subset_names = ["train-clean-100", "dev-clean"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name in subset_names:
        if name not in SUBSETS:
            print(f"Warning: Subset {name} not found. Skipping.")
            continue
            
        url = SUBSETS[name]
        tar_path = output_path / f"{name}.tar.gz"
        subset_dir = output_path / "LibriTTS" / name
        
        if not subset_dir.exists():
            if not tar_path.exists():
                download_file(url, tar_path)
            
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=output_path)
        else:
            print(f"Subset {name} already extracted.")

    root_dir = output_path / "LibriTTS"
    metadata_path = output_path / "metadata.csv"
    
    print("Gathering wav files...")
    wav_files = list(root_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} wav files.")
    
    print("Generating metadata.csv using parallel processing...")
    results = []
    with ProcessPoolExecutor() as executor:
        # Use a list to store futures and iterate with tqdm
        futures = [executor.submit(process_wav, wav, output_path) for wav in wav_files]
        for future in tqdm(futures, desc="Processing wavs"):
            res = future.result()
            if res:
                results.append(res)
    
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_id", "path", "length", "sample_rate"])
        writer.writeheader()
        writer.writerows(results)
            
    print(f"Dataset setup complete. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup LibriTTS dataset")
    parser.add_argument("--subsets", nargs="+", default=["train-clean-100", "dev-clean"], help="Subsets to download and process")
    parser.add_argument("--output_dir", default="data/libritts", help="Output directory")
    args = parser.parse_args()
    
    setup_libritts(output_dir=args.output_dir, subset_names=args.subsets)
