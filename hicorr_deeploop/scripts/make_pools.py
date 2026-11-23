r"""
Step 1 of the HiCorr + DeepLoop2 pseudobulk pipeline: one directory per group of cells.

Input: a tab-separated cell table with a header and the columns

    cell    the cell name
    group   the pool it belongs to (cell type, cluster, condition, ...)
    cis     path of the cell's HiCorr cis fragment-loop file (frag_loop.<cell>.cis.filter)
    trans   path of the cell's HiCorr trans fragment-loop file (frag_loop.<cell>.trans)

``cis`` and ``trans`` can be left out when ``--frag_dirs`` is given: each cell's files are then looked
up as <dir>/frag_loop.<cell>.cis.filter and <dir>/frag_loop.<cell>.trans in the listed directories
(first match wins; a cell found in two directories is an error unless --first_match).

Output, per group, in <out>/<group>/ (group names are made file-safe: other characters -> "_"):

    cells.tsv   cell, cis, trans of the group's cells
    cells.md5   md5 of the sorted cell table: the pipeline reruns a group from scratch when it changes
    group.txt   the original group name

and <out>/groups.tsv (group, directory, number of cells).

Usage:
    python make_pools.py --cells cells.tsv --out <dir> [--groups Alpha Beta] [--min_cells 30]
"""
import argparse
import hashlib
import os
import re
import sys

import pandas as pd


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "group"


def find_files(cell: str, dirs, first_match: bool):
    hits = [d for d in dirs if os.path.exists(os.path.join(d, f"frag_loop.{cell}.cis.filter"))]
    if not hits:
        raise FileNotFoundError(f"{cell}: no frag_loop.{cell}.cis.filter in --frag_dirs")
    if len(hits) > 1 and not first_match:
        raise ValueError(f"{cell}: found in several --frag_dirs ({', '.join(hits)}); rename or pass --first_match")
    d = hits[0]
    return os.path.join(d, f"frag_loop.{cell}.cis.filter"), os.path.join(d, f"frag_loop.{cell}.trans")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True, help="cell table (TSV with header: cell, group[, cis, trans])")
    p.add_argument("--out", required=True)
    p.add_argument("--groups", nargs="*", help="only these groups (original names; default: all)")
    p.add_argument("--frag_dirs", nargs="*", default=[], help="directories with frag_loop.<cell>.* files")
    p.add_argument("--first_match", action="store_true")
    p.add_argument("--min_cells", type=int, default=1, help="skip groups with fewer cells")
    args = p.parse_args()

    tab = pd.read_csv(args.cells, sep="\t", dtype=str)
    for c in ("cell", "group"):
        if c not in tab.columns:
            sys.exit(f"!! {args.cells}: no '{c}' column (columns: {', '.join(tab.columns)})")
    tab = tab.dropna(subset=["cell", "group"])
    if tab["cell"].duplicated().any():
        dup = tab.loc[tab["cell"].duplicated(), "cell"].iloc[:5].tolist()
        sys.exit(f"!! duplicated cells in {args.cells}, e.g. {dup}")
    if not {"cis", "trans"} <= set(tab.columns):
        if not args.frag_dirs:
            sys.exit("!! the cell table has no cis/trans columns: give them or --frag_dirs")
        files = [find_files(c, args.frag_dirs, args.first_match) for c in tab["cell"]]
        tab["cis"], tab["trans"] = [f[0] for f in files], [f[1] for f in files]
    missing = [f for f in pd.concat([tab["cis"], tab["trans"]]) if not os.path.exists(f)]
    if missing:
        sys.exit(f"!! {len(missing)} fragment files missing, e.g. {missing[:3]}")

    groups = args.groups or sorted(tab["group"].unique())
    unknown = set(groups) - set(tab["group"])
    if unknown:
        sys.exit(f"!! groups not in the table: {sorted(unknown)}")
    tags = {g: safe(g) for g in groups}
    if len(set(tags.values())) < len(tags):
        sys.exit(f"!! group names collide after making them file-safe: {tags}")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for g in groups:
        sub = tab[tab["group"] == g].sort_values("cell")[["cell", "cis", "trans"]]
        if len(sub) < args.min_cells:
            print(f"{g}: {len(sub)} cells < --min_cells {args.min_cells}, skipped")
            continue
        d = os.path.join(args.out, tags[g])
        # never write into a directory this pipeline did not make (the later steps delete their own outputs there)
        if os.path.isdir(d) and os.listdir(d) and not os.path.exists(os.path.join(d, "cells.md5")):
            sys.exit(f"!! {d} exists and was not made by this pipeline; choose another --out")
        os.makedirs(d, exist_ok=True)
        text = sub.to_csv(sep="\t", index=False)
        key = hashlib.md5(text.encode()).hexdigest()
        with open(os.path.join(d, "cells.tsv"), "w") as f:
            f.write(text)
        with open(os.path.join(d, "cells.md5"), "w") as f:
            f.write(key + "\n")
        with open(os.path.join(d, "group.txt"), "w") as f:
            f.write(str(g) + "\n")
        rows.append((g, tags[g], len(sub)))
        print(f"{g}: {len(sub)} cells -> {d} (key {key[:8]})")
    # update, not replace: groups made by earlier calls (e.g. with another --groups) stay listed
    path = os.path.join(args.out, "groups.tsv")
    new = pd.DataFrame(rows, columns=["group", "dir", "n_cells"])
    if os.path.exists(path):
        old = pd.read_csv(path, sep="\t", dtype={"group": str})
        new = pd.concat([old[~old["group"].isin(new["group"])], new])
    new.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    main()
