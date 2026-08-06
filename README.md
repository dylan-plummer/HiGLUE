# HiGLUE
Joint generative modeling of single-cell Hi-C with other single-cell modalities via Graph-Linked Unified Embedding (GLUE)

HiGLUE is an extension of [GLUE](https://scglue.readthedocs.io/en/latest/) for scHi-C and depends on the following packages: 

`pip install git+https://github.com/JinLabBioinfo/SCORE.git` (for scHi-C data processing)

`pip install git+https://github.com/dylan-plummer/GLUE.git` (custom fork of `scglue`)


After installing the above packages, you need the following input files:

* RNA data in `.h5ad` format compatible with `scanpy`

* scHi-C data in `.scool` format and metadata compatible with `SCORE`

* A `.gtf` file for the genome assembly used in the RNA and scHi-C data

Then you can run HiGLUE with the following command:

```
python higlue.py \
	--rna_file hires_brain_rna.h5ad \
	--gtf gencode.vM25.annotation.gtf \
	--train \
	--preprocess \
	SCORE \
	--dset hires_brain \
	--scool hires_brain_500kb.scool \
	--reference hires_brain_ref \
	--resolution 500kb;
```

`--preprocess` will generate all of the preprocessed data files in `data/<dset>_data/` and `--train` will train the HiGLUE model. You can provide both flags to run preprocessing and training in one command or run them separately.

Arguments provided after `SCORE` are for loading and preprocessing the scHi-C data with `SCORE` and are passed directly to `SCORE`.

## Multi-resolution training

Passing `--resolutions` trains a **single** model on Hi-C features built at several
resolutions at once. Every cell is then represented by a multi-resolution
embedding: each resolution is encoded separately (with convolutions over its band
matrix when it is too large for a dense projection) and the per-resolution
embeddings are fused into the shared latent space that is aligned with RNA.

```
python higlue.py \
	--rna_file hires_brain_rna.h5ad \
	--gtf gencode.vM25.annotation.gtf \
	--resolutions 500kb 100kb 5kb \
	--multires_strata 10 20 40 \
	--multires_max_anchors 0 0 20000 \
	--multires_anchor_subsample 4000 \
	--backed \
	--preprocess \
	--train \
	SCORE \
	--dset hires_brain \
	--scool hires_brain_5kb.scool \
	--reference hires_brain_ref \
	--resolution 5kb \
	--min_depth 100000;
```

Only one `.scool` file is needed: pass the **finest** resolution to `SCORE`
(`--scool` / `--resolution`) and the coarser grids are derived from it by
re-binning genomic coordinates. Resolutions are always ordered coarse to fine,
and per-resolution options accept either a single value or one value per
resolution:

| Option | Meaning |
| --- | --- |
| `--resolutions` | resolutions to model jointly (e.g. `500kb 100kb 5kb`) |
| `--multires_strata` | diagonal strata kept per resolution (band height); defaults to `--n_strata` |
| `--multires_max_anchors` | anchor budget per resolution (`0` = keep every detected anchor); defaults to `50000` |
| `--multires_tile_size` | anchors are kept in contiguous tiles of this many bins |
| `--multires_stat_cells` | cells scanned to rank anchors and contacts (default: all) |
| `--multires_min_frac` | minimum fraction of cells in which an anchor must be detected |
| `--multires_mlp_max_band` | bands larger than this are encoded with convolutions instead of a dense projection |
| `--multires_res_dim` | size of each per-resolution embedding (default `--h_dim`) |
| `--multires_anchor_subsample` | anchors per resolution reconstructed in each training step |
| `--multires_checkpoint` | recompute the per-resolution encoders in the backward pass (less GPU memory, more compute) |
| `--backed` | read the Hi-C matrix from disk one minibatch at a time |

After training, the fused embedding is stored in `hic.obsm["X_glue"]` and the
per-resolution parts in `hic.obsm["X_glue_<resolution>"]`.

`--use_dist_norm` and `--use_trans` apply to the pseudobulk edges of the
guidance graph. Contacts are always ranked within a stratum (i.e.
observed/expected by construction); `--use_dist_norm` additionally divides them
by the coverage of their two anchors. `--use_trans` adds top trans-chromosomal
edges, summarized at the coarsest resolution only — a dense trans matrix is
quadratic in the number of bins, and the hierarchy edges propagate the
connection down to finer resolutions.

## Embedding quality options

These apply to both single- and multi-resolution models.

| Option | Meaning |
| --- | --- |
| `--strata_weights {cell,global}` | how the library size is split across strata. `cell` (default) predicts the split from the cell embedding, so the model can represent per-cell distance-decay (cell cycle, chromatin condensation); `global` shares one profile across all cells, as before |
| `--no_strata_input_norm` | disable dividing each stratum by its typical magnitude before the encoder's log transform. Without it, the input is dominated by the first strata |
| `--lam_downsample` | weight of a depth-consistency loss: each Hi-C cell is also encoded after binomial downsampling, and the two embeddings are pulled together. This makes the embedding depth-invariant by construction rather than only discouraging depth adversarially. Costs one extra encoder pass per step |
| `--downsample_min` / `--downsample_max` | range the downsampling rate is drawn from (default 0.3–0.9) |

### Large and high resolution datasets

High resolution representations are built without ever materializing a dense
`cells x features` matrix:

* cells are streamed from the `.scool` file one at a time, both when collecting
  the statistics used for feature selection and when writing the output, so
  preprocessing memory depends on the size of the genome rather than on the
  number of cells;
* the feature space of each resolution is filtered down to its anchor budget
  *before* any per-cell data is written. Anchors are ranked by how variable
  their local contact coverage is across cells, with bins overlapping promoters
  of highly variable genes prioritized, and are kept in contiguous tiles so that
  the band matrix stays spatially coherent;
* the Hi-C dataset is written as a sparse `.h5ad`, and `--backed` keeps it on
  disk during training, densifying only the current minibatch;
* `--multires_anchor_subsample` restricts each training step's reconstruction to
  a random subset of anchors per resolution, which is what keeps the decoder
  tractable when a dataset has millions of features.

The guidance graph is also multi-scale: in addition to promoter/anchor overlaps
and the top pseudobulk contacts of each resolution, every fine anchor is linked
to the coarse bin containing it, so high resolution features remain connected to
genes even where no contact passes the loop cutoff.

Rough sizing from a HiRes mouse brain dataset (398 cells, `500kb`/`100kb`/`5kb`
with 10/20/40 strata, `20000` 5kb anchors, 1.33M Hi-C features): preprocessing
peaked at 5.8 GB of RAM and produced a 560 MB sparse `.h5ad`; training with
`--backed --multires_anchor_subsample 4000` and a batch size of 8 peaked at
1.2 GB of GPU memory.