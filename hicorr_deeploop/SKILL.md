---
name: hicorr-deeploop2
description: Run HiCorr bias correction and DeepLoop2 loop denoising on pseudobulks of single-cell Hi-C / Dip-C cells (cells pooled by cell type, cluster or condition) on the Jin lab servers, producing per-group coolers (observed, HiCorr observed/expected, DeepLoop2 denoised) and loop calls. Use when someone wants chromatin loops or loop-level contact maps for groups of scHi-C cells, asks to "run HiCorr", "run DeepLoop" or "DeepLoop2", or to make cell-type pseudobulks from per-cell fragment loops (frag_loop.<cell>.cis.filter / .trans), hg19 or mm10 DpnII.
---

# HiCorr + DeepLoop2 on scHi-C pseudobulks

This pipeline pools single cells by group and runs HiCorr and then DeepLoop2 on each pool. It is the pipeline used for
the islet cell-type loops in the HiGLUE paper (islet Dip-C, hg19, DpnII, 62–1,180 cells per type).

Everything is in `scripts/`. The driver is `scripts/run_pipeline.sh`; the other scripts are its steps.

## What to ask the user for

Ask for any missing item in one short question. Do not search the filesystem for these:

1. **The cell table**: which cells go into which pool. It is a TSV with a header and the columns `cell` and `group`.
   - Add `cis` and `trans` columns with each cell's fragment-loop file paths, **or**
   - pass `--frag_dirs` with the directories that hold `frag_loop.<cell>.cis.filter` and `frag_loop.<cell>.trans`.
2. **The genome**: `hg19` (default) or `mm10`. Both use DpnII anchors.
3. **An output directory** with room for about 35 GB per group, or about 2 GB per group with `--compact`.
4. **Which GPU** to use. Check `nvidia-smi` first; DeepLoop2 needs a few GB of GPU memory.
5. Optionally, a subset of groups or a minimum number of cells per group.

The labels can come from anything: HiGLUE predictions, clustering, FACS, or donor/condition. Example: turn a
`cell,label` CSV into a cell table.

```bash
awk -F, 'BEGIN{OFS="\t"; print "cell","group"} NR>1{print $1,$2}' labels.csv > cells.tsv
```

### Input format

Use the per-cell fragment loops from the lab's HiCorr-based mapping (for example
`.../2.mapping/processed_frag/frag_loop.<cell>.cis.filter`). They are tab-separated, with no header:

```
frag_1668   frag_12571   1   4901188      # cis: fragment 1, fragment 2, count, distance
frag_1668   frag_99012   1                # trans: fragment 1, fragment 2, count
```

The fragment numbers must come from HiCorr's DpnII fragment list for the same genome
(`<genome>.DPNII.frag.bed` in the HiCorr references). Files from any other pipeline (pairs, cool, hic) need
converting to HiCorr fragment loops first; this skill does not do that.

**Starting from raw FASTQ:** use the **hic-mapping** skill (`/mnt/jinstore/JinLab04/dmp131/skills/hic_mapping`)
with `--protocol dpnii`. It demultiplexes plates, maps the reads and writes these files to
`<mapping_out>/processed_frag/`, using the same fragment list as here. Then pass
`--frag_dirs <mapping_out>/processed_frag`. Its `frag_files.tsv` with a `group` column added also works as the
cell table.

## How to run

Use a Python environment with numpy, pandas, scipy, torch, tqdm and cooler, plus perl. For example:
`conda create -n hicorr_dl2 -c conda-forge -c pytorch python=3.10 numpy pandas scipy cooler tqdm pytorch`.

The whole run takes hours, so start it detached with its log on `/mnt` and check that it is running:

```bash
cd /mnt/.../my_project
PYTHON=$(which python) setsid nohup <skill_dir>/scripts/run_pipeline.sh \
    --cells cells.tsv --out pools --genome hg19 --gpu 0 > pools.log 2>&1 < /dev/null &
sleep 30; tail pools.log; pgrep -f run_pipeline.sh
```

Useful options (all are listed in `run_pipeline.sh --help`):
- `--groups "Alpha Beta"`
- `--min_cells 50`
- `--jobs 3`: how many HiCorr groups run in parallel. Each needs about 6 cores and, for deep pools, up to about 30 GB RAM.
- `--checkpoint <file>`: force one DeepLoop2 model for every group.
- `--n_pixels 500000`: how many of the strongest DeepLoop2 pixels are clustered into loops.
- `--compact`: delete the text intermediates once the coolers exist.
- `--steps "cooler loops"`: run only those steps.
- `--chroms "chr21 chr22"`: a quick test run.

Re-running the same command is safe. Each step is skipped when it is up to date for the group's current cells
(`cells.md5`) and DeepLoop2 checkpoint. If cells are added to a group, only that group is redone. To resume
without the cell table, run `run_pipeline.sh --out pools`.

## Steps

| Step | Script | What it does |
|---|---|---|
| pools | `make_pools.py` | `<out>/<group>/cells.tsv`, `cells.md5`, `group.txt`; `<out>/groups.tsv`. Group names are made file-safe. |
| hicorr | `merge_pool.py` + HiCorr Perl/Python | Sums the cells' fragment pairs per anchor pair (same output as HiCorr's `fragdata_to_anchordata.pl`, but memory-light). Then runs HiCorr: blacklist removal, cis ≤ 2 Mb, anchor visibility from trans reads, and an expected value by distance × anchor length. Output: `split/anchor_2_anchor.loop.chrN` (anchor1, anchor2, observed, expected). CPU only. |
| deeploop2 | DeepLoop2 `predict.py` (copied into `<out>/src/`) | Denoises each chromosome with the `loopdenoise_target` checkpoint whose training depth is closest (log scale) to the pool's cis reads ≤ 2 Mb. Input is (obs + 5) / (exp + 5); 128-anchor tiles, step 64, up to 768 anchors (~3 Mb) from the diagonal. Output: `deeploop2/chrN.denoised.anchor.to.anchor`. GPU. |
| cooler | `to_cooler.py` | Writes `<group>.hicorr.cool`, where `count` = observed and `ratio` = observed / expected, for pixels with ≥ 1 read. Writes `<group>.deeploop2.cool`, where `count` = denoised value. |
| loops | `call_loops.py` | Writes `<group>.loops.bedpe`: the top `--n_pixels` DeepLoop2 pixels at 20 kb – 2 Mb, clustered by adjacency, with one loop per cluster at its value-weighted centre. Columns include `value` (the cluster's strongest pixel), `npix`, `rel` and `shallow`. Correct the one-anchor output offset of the 35M/49M/70M/98M checkpoints first. |

The read depths are in `<out>/<group>/summary`; the checkpoint used is on line 2 of `.dl2_done`.

## Reading the outputs

- **Bins are HiCorr anchors**: variable-size DpnII anchors of about 5 kb, one bin per anchor. Bins are identical
  across groups and runs of the same genome, so pixels can be compared directly. Open with
  `cooler.Cooler(path)` and read **unbalanced**, e.g. `clr.matrix(balance=False, field="ratio")`. The data are
  already corrected, so do not balance them.
- **Comparing groups.** DeepLoop2 values depend on the checkpoint, and groups of different depth get different
  checkpoints. Do not compare raw values across groups. Compare `rel` (log2 of the value over the group's
  weakest called loop) or rank, as the paper does.
- **HiCorr ratio.** A single read at a pixel with a tiny expected value gives ratios of 50 or more. Cap the ratio
  (the paper used 10) before averaging pixels, for example in APA pile-ups.
- **Loops** are the brightest DeepLoop2 pixels, not a statistical test. The numbers are sensible for pools of a
  few M reads and up. Check them with pile-ups (APA) or against independent data before making claims.

## Depth and checkpoints

The checkpoints are `dl2_ld_1.0_<N>M_300k.pt` for N = 7, 14, 35, 49, 70, 98 and 140 M cis reads ≤ 2 Mb. They are
the DeepLoop2 group's `loopdenoise_target` models, in `$DL2_DIR/ckpts/loopdenoise_target/`.

In the islet data, the pools got:

| Group | Cells | cis reads ≤ 2 Mb | Checkpoint |
|---|---|---|---|
| PSC | 62 | 5.1 M | 7 M |
| acinar | 132 | 8.4 M | 7 M |
| ductal | 274 | 19.8 M | 14 M |
| Alpha | 1,055 | 69.7 M | 70 M |
| Beta | 1,180 | 88.8 M | 98 M |

That is about 75–80 k reads ≤ 2 Mb per cell.

- **Why match the depth:** an earlier run of the shallow pools with the 35 M checkpoint smeared signal along the
  diagonal.
- **Pools below 3.5 M reads** (half the smallest checkpoint): the run prints a warning. `call_loops.py` then calls
  loops at least 20 anchors (~100 kb) off the diagonal and sets `shallow = 1`.
- **Rough minimum:** about 50 cells of this depth per group. Fewer cells give few reliable loops.

## Time and resources

These are measured on the islet pools, on the lab server (A30 GPU):

| Step | PSC (62 cells, 5 M reads) | Beta (1,180 cells, 89 M reads) |
|---|---|---|
| HiCorr (CPU, 6 cores) | 1.3 h | 2.0 h |
| DeepLoop2, all chromosomes (GPU) | 54 min | 51 min |

**Disk:** about 20 GB of `split/` and 12 GB of `deeploop2/` text per group, regardless of depth. The coolers are
about 1.5 GB. `--compact` deletes the text once the coolers are written. A later DeepLoop2 rerun (for example
with a new checkpoint) then redoes HiCorr first.

## Notes and limits

- **mm10.** The HiCorr mm10 DpnII references exist, and the DeepLoop2 per-chromosome anchors are made from them
  (for hg19 these match the DeepLoop2 reference exactly). However, DeepLoop2 was trained on human data, so mouse
  output is untested: check it before use.
- **Arima / HindIII / hg38.** HiCorr needs a full reference set for the enzyme and genome
  (`<genome>_<enzyme>_anchors_avg.bed`, `_frag_2_anchor`, blacklist, groups, full filtered matrix, trans possible
  pairs). The lab's HiCorr reference directory has only the hg19 and mm10 DpnII sets.
- **Dependencies on the lab servers.** The run uses the lab's HiCorr scripts (`HICORR_LIB`), HiCorr references
  (`HICORR_REF`) and DeepLoop2 tree (`DL2_DIR`). The defaults point at their `/mnt/jinstore` locations, and
  environment variables override them. HiCorr's Python 2 helpers and DeepLoop2's `predict.py` / `dense_unet.py`
  are copied into `<out>/src/` on the first run, so later edits upstream do not change a project's results.
- **Reproducibility.** HiCorr's `merge_sorted_anchor_loop.pl` reads the pool's contacts and the reference full
  matrix in Perl hash order. For pixels with reads, whichever file is read last sets the distance. The two files
  use different distance conventions, so the expected values used to vary a little between runs: about 2% of
  pixels, by a median of 0.7%. `run_pipeline.sh` now fixes the hash order (`PERL_HASH_SEED=0`), so reruns are
  identical. The islet pools were made before this fix and differ from a rerun by that small amount.
- **Validation.** The skill was re-run on the islet PSC pool (62 cells), with DeepLoop2 limited to chr21–22, and
  compared with the original pipeline's output:
  - pooled contacts, read counts, visibility and trans factors: byte-identical;
  - HiCorr observed counts: identical;
  - expected values: within the variation described above;
  - DeepLoop2: r = 1.00000, and 1,999 and 2,000 of the top 2,000 pixels the same.
- **Safety.** `make_pools.py` refuses to write into an existing non-empty directory it did not create. The later
  steps only delete their own outputs inside group directories that contain `cells.md5`.
- **Failures.** Each group has `<group>/hicorr.log` and `<group>/deeploop2.<chrom>.log`. A DeepLoop2 failure
  stops the run with the log path. Fix the problem and re-run the same command; finished steps are skipped.
