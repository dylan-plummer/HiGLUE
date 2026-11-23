r"""
Step 4: HiCorr and DeepLoop2 outputs of one group -> coolers on the HiCorr anchors (variable-size bins,
one per restriction-fragment anchor, bin id = anchor number - 1, so bins match across groups).

    <group>.hicorr.cool     count = observed reads (int), ratio = observed / HiCorr expected
                            (pixels with at least one read, <= 2 Mb, cis)
    <group>.deeploop2.cool  count = DeepLoop2 denoised value (float, > 0)

Read them unbalanced: cooler.Cooler(path).matrix(balance=False, field="ratio") etc.

Usage:
    python to_cooler.py --group_dir <out>/<group> --anchors <genome>_DpnII_anchors_avg.bed --assembly hg19
"""
import argparse
import os

import cooler
import numpy as np
import pandas as pd


def read_pairs(path, cols):
    df = pd.read_csv(path, sep="\t", header=None, names=["a1", "a2"] + cols, usecols=range(2 + len(cols)))
    b1 = df["a1"].str[2:].astype(np.int64).to_numpy() - 1
    b2 = df["a2"].str[2:].astype(np.int64).to_numpy() - 1
    out = pd.DataFrame({"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2)})
    for c in cols:
        out[c] = df[c].to_numpy()
    # HiCorr's split files list every pair in both orientations: the same observed count, and for <1% of pairs a
    # slightly different expected (chr21 of the islet PSC pool: 0.7%); the first listed is kept, as in the islet analysis
    out = out.drop_duplicates(["bin1_id", "bin2_id"])
    return out.sort_values(["bin1_id", "bin2_id"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group_dir", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--assembly", default=None)
    args = p.parse_args()
    t = os.path.basename(os.path.normpath(args.group_dir))
    bins = pd.read_csv(args.anchors, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    bins["start"] -= 1  # HiCorr anchors are 1-based, closed
    chroms = list(dict.fromkeys(bins["chrom"]))

    def hicorr():
        for c in chroms:
            f = os.path.join(args.group_dir, "split", f"anchor_2_anchor.loop.{c}")
            if not os.path.exists(f):
                continue
            df = read_pairs(f, ["obs", "exp"])
            df = df[df["obs"] > 0]
            yield pd.DataFrame({"bin1_id": df["bin1_id"], "bin2_id": df["bin2_id"], "count": df["obs"].astype(np.int32),
                                "ratio": (df["obs"] / df["exp"]).astype(np.float32)})

    def deeploop2():
        for c in chroms:
            f = os.path.join(args.group_dir, "deeploop2", f"{c}.denoised.anchor.to.anchor")
            if not os.path.exists(f):
                continue
            df = read_pairs(f, ["count"])
            df = df[df["count"] > 0]
            yield df.astype({"count": np.float32})

    for name, it, kw in ((f"{t}.hicorr.cool", hicorr(), dict(columns=["count", "ratio"],
                                                             dtypes={"count": np.int32, "ratio": np.float32})),
                         (f"{t}.deeploop2.cool", deeploop2(), dict(dtypes={"count": np.float32}))):
        out = os.path.join(args.group_dir, name)
        tmp = out + ".tmp"
        cooler.create_cooler(tmp, bins, it, ordered=True, assembly=args.assembly, **kw)
        os.replace(tmp, out)
        print(f"{t}: wrote {out}")


if __name__ == "__main__":
    main()
