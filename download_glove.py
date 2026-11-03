import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
TARGET_DIR = Path(__file__).resolve().parent / "data" / "glove"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_glove(target_dir=TARGET_DIR):
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "glove.6B.zip"
    txt_path = target_dir / "glove.6B.300d.txt"

    if txt_path.exists():
        print(f"GloVe 300d embeddings already exist at: {txt_path}")
        return txt_path

    print(f"Downloading GloVe embeddings from {GLOVE_URL} (~862 MB)...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="glove.6B.zip") as t:
        urllib.request.urlretrieve(GLOVE_URL, filename=zip_path, reporthook=t.update_to)

    print("Extracting glove.6B.300d.txt...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("glove.6B.300d.txt", target_dir)

    print(f"GloVe embeddings ready at: {txt_path}")
    return txt_path

if __name__ == "__main__":
    download_and_extract_glove()
