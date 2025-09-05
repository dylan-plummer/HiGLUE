# HiGLUE
Joint generative modeling of single-cell Hi-C with other single-cell modalities via Graph-Linked Unified Embedding (GLUE)

HiGLUE depends on the following packages: 

* `SCORE`: https://github.com/JinLabBioinfo/SCORE.git

* `scglue`: https://scglue.readthedocs.io/en/latest/ (note that HiGLUE currently needs a custom fork of `scglue`)

To install these packages for HiGLUE, you can run the following:

`pip install git+https://github.com/JinLabBioinfo/SCORE.git`

`pip install git+https://github.com/dylan-plummer/GLUE.git`


After installing the above packages, you need the following input files:

* RNA data in `h5ad` format compatible with `scanpy`

* scHi-C data in `.scool` format and metadata compatible with `SCORE`

* A `gtf` file for the genome assembly used in the RNA and scHi-C data

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