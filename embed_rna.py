import sys
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

rna_file = sys.argv[1]
rna = ad.read_h5ad(rna_file)

rna.X = rna.layers["counts"]
print(rna)

sc.pp.filter_genes(rna, min_counts=1)

print('Embedding RNA...')
sc.pp.highly_variable_genes(rna, n_top_genes=10000, flavor="seurat_v3")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")
sc.pp.neighbors(rna, n_pcs=100, metric="cosine")
sc.tl.umap(rna)

fig = sc.pl.umap(rna, color=["celltype"], return_fig=True, wspace=0.6)
fig.tight_layout()
fig.savefig("rna_umap.png")
plt.close()