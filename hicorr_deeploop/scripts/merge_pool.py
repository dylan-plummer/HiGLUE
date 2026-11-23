r"""
Step 2a: sums the fragment-pair counts of a group's cells per anchor pair (the HiCorr frag -> anchor map)
and writes <group_dir>/end_loop.cis and end_loop.trans exactly like HiCorr's fragdata_to_anchordata.pl:
pairs inside one anchor dropped, rows sorted by anchor 1 then anchor 2, orientation as in the fragment
files. Unlike the Perl script it never holds every pair in a hash, so thousands of cells fit in memory.

Fragment files: tab-separated, no header, columns frag_<i> frag_<j> count [...] (HiCorr's per-cell
frag_loop.<cell>.cis.filter / .trans).

Usage:
    python merge_pool.py --group_dir <out>/<group> --frag2anchor <genome>_DpnII_frag_2_anchor [--threads 6]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

LUT = None
SHIFT = 20  # anchor ids are packed as a1 << SHIFT | a2


def frag_map(path):
    m = pd.read_csv(path, sep="\t", header=None, names=["f", "a"], dtype=str)
    fid = m["f"].str[5:].astype(np.int64).to_numpy()  # frag_<i>
    aid = m["a"].str[2:].astype(np.int64).to_numpy()  # A_<j>
    lut = np.zeros(fid.max() + 1, dtype=np.int64)
    lut[fid] = aid
    return lut


def _init(path):
    global LUT
    LUT = frag_map(path)


def _read_cell(path):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["f1", "f2", "n"], dtype=str)
    if len(df) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a1 = LUT[df["f1"].str[5:].astype(np.int64).to_numpy()]
    a2 = LUT[df["f2"].str[5:].astype(np.int64).to_numpy()]
    n = df["n"].astype(np.int64).to_numpy()
    keep = a1 != a2
    return (a1[keep] << SHIFT) | a2[keep], n[keep]


def _aggregate(keys, counts):
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=counts).astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group_dir", required=True)
    p.add_argument("--frag2anchor", required=True)
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()
    cells = pd.read_csv(os.path.join(args.group_dir, "cells.tsv"), sep="\t")
    if frag_map(args.frag2anchor).max() >= (1 << SHIFT):
        raise SystemExit("!! more anchors than the packing allows (2^20); raise SHIFT")
    with Pool(args.threads, initializer=_init, initargs=(args.frag2anchor,)) as pool:
        for kind in ("cis", "trans"):
            keys, counts = np.zeros(0, np.int64), np.zeros(0, np.int64)
            buf_k, buf_n = [], []
            for i, (k, n) in enumerate(pool.imap(_read_cell, cells[kind], chunksize=4)):
                buf_k.append(k)
                buf_n.append(n)
                if len(buf_k) == 100 or i == len(cells) - 1:
                    keys, counts = _aggregate(np.concatenate([keys] + buf_k), np.concatenate([counts] + buf_n))
                    buf_k, buf_n = [], []
            out = os.path.join(args.group_dir, f"end_loop.{kind}")
            # np.unique sorts by (a1 << SHIFT | a2), i.e. by anchor 1 then anchor 2 like the Perl script
            pd.DataFrame({"a1": "A_" + pd.Series(keys >> SHIFT).astype(str),
                          "a2": "A_" + pd.Series(keys & ((1 << SHIFT) - 1)).astype(str),
                          "n": counts}).to_csv(out, sep="\t", header=False, index=False)
            print(f"{out}: {len(keys)} anchor pairs, {counts.sum()} contacts")


if __name__ == "__main__":
    main()
