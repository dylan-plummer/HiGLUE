# hicorr-deeploop2 (Claude Code skill)

HiCorr + DeepLoop2 on pseudobulks of single-cell Hi-C / Dip-C cells grouped by cell type, cluster or condition.
For each group it produces:

- a HiCorr cooler (observed and observed/expected),
- a DeepLoop2 cooler (denoised),
- loop calls.

It runs on the Jin lab servers. It is the pipeline behind the islet cell-type loops in the HiGLUE paper.

## Install

Link the shared copy into your Claude Code skills folder, together with the mapping skill that makes its inputs
from raw FASTQ:

```bash
mkdir -p ~/.claude/skills
ln -s /mnt/jinstore/JinLab04/dmp131/skills/hicorr_deeploop ~/.claude/skills/hicorr-deeploop2
ln -s /mnt/jinstore/JinLab04/dmp131/skills/hic_mapping ~/.claude/skills/hic-mapping
```

Then ask Claude, for example: "run HiCorr and DeepLoop2 on my microglia and astrocyte pools, cell table is
`/mnt/.../cells.tsv`". Claude will ask for anything missing (genome, output directory, GPU) and start the run.

## Run it yourself

```bash
PYTHON=$(which python) scripts/run_pipeline.sh --cells cells.tsv --out pools --genome hg19 --gpu 0
scripts/run_pipeline.sh --help
```

`cells.tsv` has the columns `cell`, `group`, `cis` and `trans`. `cis` and `trans` are the paths of each cell's
`frag_loop.<cell>.cis.filter` and `.trans` files; leave them out and pass `--frag_dirs` instead to have them
looked up.

See `SKILL.md` for:

- the outputs and how to read them,
- depth and checkpoint choices,
- run times (about 1.5–2 h of CPU and 1 h of GPU per group),
- disk use (about 35 GB per group, or 2 GB with `--compact`).
