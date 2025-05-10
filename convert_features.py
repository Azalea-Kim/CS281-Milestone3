#!/usr/bin/env python3
"""
Convert `features.h5`
to COLMAP’s per‑image text format:

NUM_FEATURES 128
X Y SCALE ORIENTATION D1 … D128   (one line per feature)

Usage
-----
python h5_features_to_txt.py \
        --h5   path/to/features.h5 \
        --imgs path/to/images_folder
"""

import h5py, argparse, pathlib, numpy as np
from tqdm import tqdm

def save_feature_txt(out_path, kpts, desc):
    """Write a single image’s features to COLMAP text format."""
    with open(out_path, "w") as f:
        f.write(f"{len(kpts)} 128\n")
        # SuperPoint gives only X,Y; we invent SCALE=1, ORI=0
        scale = 1.0
        ori   = 0.0
        for (x, y), d in zip(kpts, desc):
            d_int = np.clip(np.rint((d + 1.0) * 127.5), 0, 255).astype(int)
            f.write(f"{x:.6f} {y:.6f} {scale:.3f} {ori:.3f} " +
                    " ".join(map(str, d_int)) + "\n")

def main(h5_path, img_dir):
    img_dir = pathlib.Path(img_dir)
    with h5py.File(h5_path, "r") as f:
        # descend a wrapper group (e.g. 'images_2') if necessary
        if len(f.keys()) == 1 and isinstance(f[list(f.keys())[0]], h5py.Group):
            f = f[list(f.keys())[0]]

        for img_path in tqdm(sorted(img_dir.glob("*")), desc="Images"):
            name = img_path.name
            if name not in f:
                print(f"[!] no features for {name}, skipping")
                continue
            g   = f[name]
            kpt = np.asarray(g["keypoints"], np.float32)
            desc= np.asarray(g["descriptors"], np.float32)
            # ensure (N,128)
            if desc.shape[0] == 128: desc = desc.T
            save_feature_txt(img_path.with_suffix(img_path.suffix + ".txt"), kpt, desc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",   required=True, help="features.h5")
    ap.add_argument("--imgs", required=True, help="image folder")
    args = ap.parse_args()
    main(args.h5, args.imgs)
