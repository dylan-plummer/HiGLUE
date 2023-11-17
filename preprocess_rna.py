import pandas as pd
import anndata as ad 
import scanpy as sc

rna_file = '/mnt/jinstore/JinLab01/dmp131/scRNA/raw_matrices/islet.rna.matrix'
rna_ref = '/mnt/jinstore/JinLab01/dmp131/scRNA/raw_matrices/islet.rna.metadata'

rna_df = pd.read_csv(rna_file, sep='\t', index_col=0)
rna_meta = pd.read_csv(rna_ref, sep='\t', index_col=0)
cellnames = rna_df.index
x = rna_df.to_numpy()
rna = ad.AnnData(x)
rna.obs_names = cellnames
rna.var_names = rna_df.columns

rna.obs['celltype'] = rna_meta.loc[cellnames, 'Cell_type']
rna.obs['batch'] = rna_meta.loc[cellnames, 'Donor']
rna.layers["counts"] = rna.X.copy()

rna.write('rna.h5ad', compression='gzip')