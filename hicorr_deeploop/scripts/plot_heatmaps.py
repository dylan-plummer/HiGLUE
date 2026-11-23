r"""
Step 6: test heatmaps, a quick visual check of every run.

The same regions are drawn for every group, so groups can be compared by eye:

    <out>/heatmap_regions.tsv               the regions (chrom, start, end, picked_from); reused on reruns
    <out>/<group>/<group>.heatmaps.png      one row per region: HiCorr observed/expected (raw pixels, capped at 10)
                                            | DeepLoop2,
                                            with the group's called loops marked in the upper triangle
    <out>/heatmaps.deeploop2.png            one row per region, one column per group: DeepLoop2 on a colour
                                            scale shared across groups (per region)

Regions: given with --regions "chr7:129000000-130500000 chr17:...", or picked automatically: windows of
--width (1.5 Mb by default, inside the 2 Mb HiCorr band), centred on the strongest loops 200-800 kb long, on
different chromosomes, taken in turn from each group's loops so no single group decides what is shown.
Delete heatmap_regions.tsv (or pass --regions) to pick new ones.

Axes are HiCorr anchors (~5 kb each, variable size), labelled with genomic positions. DeepLoop2 values are only
comparable between groups denoised with the same checkpoint; the column titles name each group's checkpoint.

Usage:
    python plot_heatmaps.py --out <out> --groups G1 G2 ... [--regions "..."] [--n_regions 4] [--width 1500000]
"""
import argparse
import os
import re

import cooler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RATIO_CAP = 10  # as in the islet analysis; single reads over tiny expected values reach 50+


def parse_region(s):
    m = re.fullmatch(r"(\w+):([\d,]+)-([\d,]+)", s)
    if not m:
        raise SystemExit(f"bad region {s!r}: use chrom:start-end")
    return m.group(1), int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))


def pick_regions(out, groups, n, width):
    loops = {}
    for g in groups:
        f = os.path.join(out, g, f"{g}.loops.bedpe")
        if os.path.exists(f):
            l = pd.read_csv(f, sep="\t")
            d = (l.start2 + l.end2) / 2 - (l.start1 + l.end1) / 2
            loops[g] = l[(d >= 200_000) & (d <= 800_000) & (l.chrom1 == l.chrom2)].sort_values("rank")
    if not loops:
        raise SystemExit("no loops.bedpe in any group: run the loops step first, or pass --regions")
    picked, used, pos = [], set(), {g: 0 for g in loops}
    order = [g for g in groups if g in loops]
    while len(picked) < n and any(pos[g] < len(loops[g]) for g in order):
        for g in order:
            l = loops[g]
            while pos[g] < len(l) and l.chrom1.iat[pos[g]] in used:
                pos[g] += 1
            if pos[g] >= len(l) or len(picked) == n:
                continue
            r = l.iloc[pos[g]]
            centre = int((r.start1 + r.end2) / 2)
            start = max(0, centre - width // 2)
            picked.append((r.chrom1, start, start + width, g))
            used.add(r.chrom1)
    return pd.DataFrame(picked, columns=["chrom", "start", "end", "picked_from"])


def ticks(ax, bins, n=4):
    k = np.linspace(0, len(bins) - 1, n).astype(int)
    lab = [f"{bins.start.iat[i] / 1e6:.2f}" for i in k]
    ax.set_xticks(k, lab, fontsize=7)
    ax.set_yticks(k, lab, fontsize=7)


def window(clr, region, field):
    bins = clr.bins().fetch(region)[["chrom", "start", "end"]].reset_index()
    m = clr.matrix(balance=False, field=field).fetch(region).astype(float)
    return bins, m


def vmax_of(mats, q=98, zeros=False):
    """q-th percentile of the off-diagonal (>= 4 anchors) values, pooled: shows loops without the diagonal.
    zeros=True keeps empty pixels, for the sparse HiCorr matrix of shallow pools."""
    v = []
    for m in mats:
        iu = np.triu_indices_from(m, k=4)
        x = m[iu]
        v.append(x[np.isfinite(x) & ((x > 0) | zeros)])
    v = np.concatenate(v) if v else np.array([1.0])
    return max(float(np.percentile(v, q)), 1e-6) if v.size else 1.0


def checkpoint_of(gdir):
    f = os.path.join(gdir, ".dl2_done")
    if not os.path.exists(f):
        return "?"
    m = re.search(r"_(\d+M)_", open(f).read().split()[1])
    return m.group(1) if m else "?"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--groups", nargs="+", required=True, help="group directory names")
    p.add_argument("--regions", default="", help='"chrom:start-end ..." (default: picked from the loops)')
    p.add_argument("--n_regions", type=int, default=4)
    p.add_argument("--width", type=int, default=1_500_000)
    args = p.parse_args()
    out, groups = args.out, args.groups

    reg_file = os.path.join(out, "heatmap_regions.tsv")
    if args.regions:
        regs = pd.DataFrame([(*parse_region(s), "--regions") for s in args.regions.split()],
                            columns=["chrom", "start", "end", "picked_from"])
    elif os.path.exists(reg_file):
        regs = pd.read_csv(reg_file, sep="\t")
    else:
        regs = pick_regions(out, groups, args.n_regions, args.width)
    regs.to_csv(reg_file, sep="\t", index=False)

    dl = {g: cooler.Cooler(os.path.join(out, g, f"{g}.deeploop2.cool")) for g in groups}
    hc = {g: cooler.Cooler(os.path.join(out, g, f"{g}.hicorr.cool")) for g in groups}
    loops = {}
    for g in groups:
        f = os.path.join(out, g, f"{g}.loops.bedpe")
        loops[g] = pd.read_csv(f, sep="\t") if os.path.exists(f) else None
    ck = {g: checkpoint_of(os.path.join(out, g)) for g in groups}

    data = []  # per region: bins, {g: (hicorr ratio, deeploop2)}
    for r in regs.itertuples(index=False):
        region = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
        per = {}
        for g in groups:
            bins, d = window(dl[g], region, "count")
            _, h = window(hc[g], region, "ratio")
            per[g] = (np.clip(np.nan_to_num(h, nan=0, posinf=RATIO_CAP), 0, RATIO_CAP), d)   # raw, unsmoothed
        data.append((region, bins, per, vmax_of([per[g][1] for g in groups])))

    nr = len(data)
    # ---- per group: HiCorr | DeepLoop2 (+ loops)
    for g in groups:
        fig, axes = plt.subplots(nr, 2, figsize=(8, 4 * nr), squeeze=False)
        for i, (region, bins, per, vmax) in enumerate(data):
            h, d = per[g]
            for j, (m, title, vm) in enumerate(((h, f"HiCorr obs/exp (raw, cap {RATIO_CAP})", vmax_of([h])),
                                                (d, f"DeepLoop2 ({ck[g]} ckpt)", vmax))):
                ax = axes[i, j]
                im = ax.imshow(m, cmap="Reds", vmin=0, vmax=vm, interpolation="none")
                ax.set_title(f"{region}\n{title}", fontsize=8)
                ticks(ax, bins)
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
            l = loops[g]
            if l is not None:
                idx = {(c, s): k for k, (c, s) in enumerate(zip(bins.chrom, bins.start))}
                k1 = [idx.get((c, s)) for c, s in zip(l.chrom1, l.start1)]
                k2 = [idx.get((c, s)) for c, s in zip(l.chrom2, l.start2)]
                xy = [(a, b) for a, b in zip(k1, k2) if a is not None and b is not None]
                if xy:
                    a, b = np.array(xy).T
                    axes[i, 1].scatter(np.maximum(a, b), np.minimum(a, b), s=12, facecolors="none",
                                       edgecolors="tab:blue", linewidths=0.6)
                axes[i, 1].set_xlabel(f"{len(xy)} called loops (circles, upper triangle); Mb", fontsize=7)
        fig.suptitle(g, fontsize=10)
        fig.tight_layout()
        f = os.path.join(out, g, f"{g}.heatmaps.png")
        fig.savefig(f, dpi=110)
        plt.close(fig)
        print(f"{g}: wrote {f}")

    # ---- all groups side by side, DeepLoop2 on a shared scale per region
    ng = len(groups)
    fig, axes = plt.subplots(nr, ng, figsize=(3.6 * ng + 0.6, 3.6 * nr), squeeze=False)
    for i, (region, bins, per, vmax) in enumerate(data):
        for j, g in enumerate(groups):
            ax = axes[i, j]
            im = ax.imshow(per[g][1], cmap="Reds", vmin=0, vmax=vmax, interpolation="none")
            ax.set_title(f"{g} ({ck[g]} ckpt)\n{region}", fontsize=8)
            ticks(ax, bins)
        fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.02, pad=0.01).ax.tick_params(labelsize=7)
    if len(set(ck.values())) > 1:
        fig.suptitle("checkpoints differ between groups: compare patterns, not values", fontsize=9)
    f = os.path.join(out, "heatmaps.deeploop2.png")
    fig.savefig(f, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f} ({nr} regions x {ng} groups; regions in {reg_file})")


if __name__ == "__main__":
    main()
