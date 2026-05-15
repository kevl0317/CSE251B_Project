"""
Prepares the FineWeb-Edu dataset (build-nanogpt/edu_fineweb10B) into
train.bin / val.bin binary files for nanoGPT training.
"""

import os
import numpy as np
from tqdm import tqdm

try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ not defined (e.g. Jupyter / python -c)
    _script_dir = os.getcwd()

DATA_DIR = os.path.join(_script_dir, "..", "..", "..", "build-nanogpt", "edu_fineweb10B")
OUT_DIR  = _script_dir

if __name__ == '__main__':
    shards = {}
    for split in ["val", "train"]:
        files = sorted(os.listdir(DATA_DIR))
        files = [f for f in files if f.startswith(f"edufineweb_{split}_") and f.endswith(".npy")]
        shards[split] = files
        print(f"{split}: {len(files)} shards found")

    # Preview
    test = np.load(os.path.join(DATA_DIR, shards["val"][0]))
    print(f"Val shard shape: {test.shape}, dtype: {test.dtype}")
    print(f"First 20 tokens: {test[:20]}")

    # Write shards to binary files
    for split, files in shards.items():
        arr_lens = []
        for f in tqdm(files, desc=f"scanning {split}"):
            shard = np.load(os.path.join(DATA_DIR, f))
            arr_lens.append(len(shard))

        total_len = sum(arr_lens)
        out_path = os.path.join(OUT_DIR, f"{split}.bin")
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_len,))

        idx = 0
        for f in tqdm(files, desc=f"writing {split}"):
            shard = np.load(os.path.join(DATA_DIR, f))
            arr[idx : idx + len(shard)] = shard
            idx += len(shard)
        arr.flush()
        print(f"Wrote {out_path}  ({total_len:,} tokens, {os.path.getsize(out_path)/1e9:.2f} GB)")

    # Verify
    for split in ["val", "train"]:
        m = np.memmap(os.path.join(OUT_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
        print(f"{split}.bin  →  {len(m):,} tokens  |  first 10: {m[:10]}")