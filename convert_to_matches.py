"""
Convert `matches.h5`
to COLMAP’s matches format:
"""
import h5py, argparse, pathlib, numpy as np, textwrap
from tqdm import tqdm

def main(h5_path, out_txt):
    out_txt = pathlib.Path(out_txt)
    with h5py.File(h5_path, "r") as f, out_txt.open("w") as fout:
        for g1 in tqdm(f.keys(), desc="Pairs"):
            for g2 in f[g1]:
                n1 = g1.split("-",1)[-1]
                n2 = g2.split("-",1)[-1]
                m  = np.asarray(f[g1][g2]["matches0"], np.int32)
                idx = np.where(m >= 0)[0]
                if idx.size == 0:           # skip pairs with no matches
                    continue
                fout.write(f"{n1} {n2}\n")
                for i,j in zip(idx, m[idx]):
                    fout.write(f"{i} {j}\n")
                fout.write("\n")            # blank line between pairs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",  required=True, help="matches.h5")
    ap.add_argument("--out", required=True, help="output txt file")
    args = ap.parse_args()
    main(args.h5, args.out)
