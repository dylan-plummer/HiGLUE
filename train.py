import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
from matplotlib import rcParams
import os
import sys
import wandb
import json
import argparse
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna_file', type=str)
    parser.add_argument('--hic_file', type=str)
    parser.add_argument('--graph_file', type=str)
    parser.add_argument('--prior', type=str, default='dcq')
    parser.add_argument('--dset', '--dataset_name', type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--hic_dist', type=str, default='NB')
    parser.add_argument('--n_neighbors', type=int, default=10)

    args = parser.parse_args()

    rna = ad.read_h5ad(args.rna_file)
    hic = ad.read_h5ad(args.hic_file)

    prior_name = args.prior
    use_rep = None
    use_batch = None
    use_celltype = False
    use_gene_scores = False 
    use_wandb = args.wandb

    if use_wandb:
        # Just start a W&B run, passing `sync_tensorboard=True)`, to plot your Tensorboard files
        wandb.init(project=f'GLUE-{args.dset}-hic-prior', sync_tensorboard=True, config={'prior': prior_name})

    prior = nx.read_graphml(args.graph_file)

    rna.var["highly_variable"] = rna.var["highly_variable"] & rna.var["in_hic"]
    hic.var["highly_variable"] = hic.var[f"{prior_name}_highly_variable"]


    scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_rep="X_pca", use_layer="counts", use_cell_type="celltype" if use_celltype else None, use_batch=use_batch)
    if use_rep is not None:
        scglue.models.configure_dataset(hic, "NB", use_highly_variable=True, use_rep=use_rep, use_layer="counts", use_batch="batch")
    else:
        scglue.models.configure_dataset(hic, "NB", use_highly_variable=True, use_layer="counts", use_batch="batch")

    print(rna.var.query("highly_variable"))
    print(hic.var.query("highly_variable"))

    print(f"Total nodes in prior: {len(prior.nodes)}")

    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "hic": hic}, prior,
        #init_kws={"use_node_attributes": node_attrs},
        fit_kws={"directory": "glue"}
    )

    glue.save(f"glue_hic_{prior_name}_prior.dill")

    dx = scglue.models.integration_consistency(
        glue, {"rna": rna, "hic": hic}, prior
    )
    print(dx)
    if use_wandb:
        c_table = wandb.Table(dataframe=dx)
        wandb.log({"consistency_table": c_table})
        for i, row in dx.iterrows():
            wandb.log({"n_meta": row['n_meta'], "consistency": row['consistency']})
    sns.lineplot(x="n_meta", y="consistency", data=dx).axhline(y=0.05, c="darkred", ls="--")
    plt.savefig(f'{args.dset}_consistency.png')
    plt.close()

    # embed and visualize
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    hic.obsm["X_glue"] = glue.encode_data("hic", hic)

    rna.obs['domain'] = 'rna'
    hic.obs['domain'] = 'hic'

    rna.obs['old_celltype'] = rna.obs['celltype']
    hic.obs['old_celltype'] = hic.obs['celltype']
    scglue.data.transfer_labels(rna, hic, "celltype", use_rep="X_glue", n_neighbors=args.n_neighbors)

    combined = ad.concat([rna, hic])

    sc.pp.neighbors(hic, use_rep="X_glue", metric="cosine")
    sc.tl.umap(hic)
    fig = sc.pl.umap(hic, color=["old_celltype", "celltype", "celltype_confidence"], wspace=0.4, return_fig=True)
    fig.savefig('glue_umap.png')
    plt.close()
    if use_wandb:
        wandb.log({"hic_umap": wandb.Image('glue_umap.png')})

    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine", n_neighbors=args.n_neighbors)
    sc.tl.umap(combined)
    fig = sc.pl.umap(combined, color=["old_celltype", "celltype", "domain"], wspace=0.4, return_fig=True)
    plt.tight_layout()
    fig.savefig('glue_joint_umap.png')
    plt.close()
    if use_wandb:
        wandb.log({"joint_umap": wandb.Image('glue_joint_umap.png')})

    fig = sc.pl.umap(combined[combined.obs["domain"] == "hic"], color=["old_celltype", "celltype"], wspace=0.25, return_fig=True)
    fig.savefig('glue_joint_umap_hic.png')
    plt.close()
    if use_wandb:
        wandb.log({"joint_umap_hic": wandb.Image('glue_joint_umap_hic.png')})

    fig = sc.pl.umap(combined[combined.obs["domain"] == "rna"], color=["celltype"], wspace=0.25, return_fig=True)
    fig.savefig('glue_joint_umap_rna.png')
    plt.close()
    if use_wandb:
        wandb.log({"joint_umap_rna": wandb.Image('glue_joint_umap_rna.png')})