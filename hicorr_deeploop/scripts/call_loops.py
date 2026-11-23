r"""
Step 5 (optional): loops of one group from its DeepLoop2 cooler, as in the islet pseudobulk analysis.

Loops (``--caller cluster``, default; loop_caller.py): the ``--n_pixels`` (500,000) strongest DeepLoop2 pixels
between 20 kb and 2 Mb genome-wide, clustered by adjacency (8-connected on the anchor grid), one loop per cluster at
its value-weighted centre of mass; ``value`` = the cluster's strongest pixel, ``npix`` = its size, ``stripe`` = calls
sharing the busier anchor (+-1 anchor). ``--caller peak``: DeepLoop2 peaks (log2 prominence >= ``--min_prom``,
summits within ``--merge_bp`` merged) filled up to ``--n_loops`` with pixels ``--fill_bp`` apart (tiles stripes).
``--caller pixel``: the ``--n_loops`` strongest pixels after greedy non-maximum suppression (no two loops within
2 anchors in both dimensions).

Note: the 35M, 49M, 70M and 98M loopdenoise_target checkpoints return their output one anchor down-right of the input
(both axes); correct the DeepLoop2 cooler first (islet dl2_ckpt_test/make_shiftfix.py) when a pool used one of them.

Shallow-pool safeguard: when a pool has less than half the read depth of the DeepLoop2 checkpoint
it was denoised with (only possible below the smallest checkpoint, 7 M), DeepLoop2 smears signal along
the diagonal; its loops are then called at least ``--smear_off`` anchors (~100 kb) off the diagonal and
``shallow`` is set in the output.

DeepLoop2 values are not comparable between checkpoints, so compare groups by ``rel`` =
log2(value / the group's weakest called loop), not by raw value.

Output <group_dir>/<group>.loops.bedpe (tab-separated, header):
    chrom1 start1 end1 chrom2 start2 end2 value rel rank shallow [kind stripe npix]

Usage:
    python call_loops.py --group_dir <out>/<group> [--n_pixels 500000] [--caller cluster|peak|pixel]
"""
import argparse
import os
import re

import sys

import cooler
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_caller import call_loops_clusters, call_loops_peaks  # noqa: E402

MISMATCH = 0.5


def call_loops(dl, n, min_d, max_d, min_off=0):
    bins = dl.bins()[["chrom", "start", "end"]][:]
    mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
    cand = []
    for c in dl.chromnames:
        px = dl.pixels().fetch(c)
        if px.empty:
            continue
        d = mid[px["bin2_id"].to_numpy()] - mid[px["bin1_id"].to_numpy()]
        off = px["bin2_id"].to_numpy() - px["bin1_id"].to_numpy()
        px = px[(d >= min_d) & (d <= max_d) & (off >= min_off)]
        cand.append(px.nlargest(3 * n, "count"))
    cand = pd.concat(cand).sort_values("count", ascending=False)
    taken, keep = set(), []
    for b1, b2, v in zip(cand["bin1_id"].to_numpy(), cand["bin2_id"].to_numpy(), cand["count"].to_numpy()):
        if (b1, b2) in taken:
            continue
        keep.append((b1, b2, v))
        for i in range(-2, 3):
            for j in range(-2, 3):
                taken.add((b1 + i, b2 + j))
        if len(keep) == n:
            break
    k = pd.DataFrame(keep, columns=["bin1", "bin2", "value"])
    b1, b2 = bins.iloc[k["bin1"]].reset_index(drop=True), bins.iloc[k["bin2"]].reset_index(drop=True)
    return pd.DataFrame({"chrom1": b1["chrom"], "start1": b1["start"], "end1": b1["end"],
                         "chrom2": b2["chrom"], "start2": b2["start"], "end2": b2["end"], "value": k["value"]})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group_dir", required=True)
    p.add_argument("--n_loops", type=int, default=75_000)
    p.add_argument("--caller", choices=["cluster", "peak", "pixel"], default="cluster")
    p.add_argument("--n_pixels", type=int, default=500_000)
    p.add_argument("--min_prom", type=float, default=0.5)
    p.add_argument("--merge_bp", type=int, default=20_000)
    p.add_argument("--fill_bp", type=int, default=15_000)
    p.add_argument("--min_dist", type=int, default=20_000)
    p.add_argument("--max_dist", type=int, default=2_000_000)
    p.add_argument("--smear_off", type=int, default=20)
    args = p.parse_args()
    t = os.path.basename(os.path.normpath(args.group_dir))
    dl = cooler.Cooler(os.path.join(args.group_dir, f"{t}.deeploop2.cool"))
    # .dl2_done: cells key, checkpoint file name, cis reads <= 2 Mb
    _, model, reads = open(os.path.join(args.group_dir, ".dl2_done")).read().split()[:3]
    depth = float(re.search(r"_(\d+)M_", model).group(1)) * 1e6
    shallow = float(reads) < MISMATCH * depth
    if args.caller in ("cluster", "peak"):
        import loop_caller
        loop_caller.MIN_D, loop_caller.MAX_D = args.min_dist, args.max_dist
        if args.caller == "cluster":
            k = call_loops_clusters(dl, args.n_pixels, args.smear_off if shallow else 0)
        else:
            k = call_loops_peaks(dl, args.n_loops, args.smear_off if shallow else 0, args.min_prom, args.merge_bp,
                                 fill_bp=args.fill_bp)
            k["npix"] = 1
        bins = dl.bins()[["chrom", "start", "end"]][:]
        b1, b2 = bins.iloc[k["bin1"]].reset_index(drop=True), bins.iloc[k["bin2"]].reset_index(drop=True)
        loops = pd.DataFrame({"chrom1": b1["chrom"], "start1": b1["start"], "end1": b1["end"],
                              "chrom2": b2["chrom"], "start2": b2["start"], "end2": b2["end"], "value": k["value"]})
        extra = k[["kind", "stripe", "npix"]]
    else:
        loops = call_loops(dl, args.n_loops, args.min_dist, args.max_dist, args.smear_off if shallow else 0)
        extra = None
    if extra is not None:  # rank by value across peaks and fills
        loops = pd.concat([loops, extra.reset_index(drop=True)], axis=1).sort_values("value", ascending=False)
        loops = loops.reset_index(drop=True)
        extra = loops[["kind", "stripe", "npix"]]
        loops = loops.drop(columns=["kind", "stripe", "npix"])
    thr = loops["value"].min()
    loops["rel"] = np.log2(loops["value"] / thr).round(4)
    loops["rank"] = np.arange(1, len(loops) + 1)
    loops["shallow"] = int(shallow)
    if extra is not None:
        for col in ("kind", "stripe", "npix"):
            loops[col] = extra[col].to_numpy()
    out = os.path.join(args.group_dir, f"{t}.loops.bedpe")
    loops.to_csv(out, sep="\t", index=False)
    print(f"{t}: {len(loops)} loops -> {out} (threshold {thr:.4g}, checkpoint {model}, {float(reads) / 1e6:.1f} M reads"
          f"{', SHALLOW: loops >= %d anchors off the diagonal' % args.smear_off if shallow else ''})")


if __name__ == "__main__":
    main()
