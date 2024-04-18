import os
import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import cooler
import cooltools
from cooler._logging import set_verbosity_level
from matplotlib import rcParams
import sys
import math
import itertools
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, adjusted_rand_score
from networkx.algorithms.bipartite import biadjacency_matrix
from score.sc_args import parse_args
from score.utils.utils import anchor_to_locus, anchor_list_to_dict, sorted_nicely
from preprocess_data import preprocess_higlue

if __name__ == '__main__':
    glue_parser = argparse.ArgumentParser()
    # preprocessing args
    glue_parser.add_argument('--data_dir', type=str, default='')
    glue_parser.add_argument('--loop_q', type=str, default='0.99')
    glue_parser.add_argument('--n_strata', type=int, default=5)
    glue_parser.add_argument('--use_ice', action='store_true')
    glue_parser.add_argument('--use_2d', action='store_true')
    glue_parser.add_argument('--viz_rna', action='store_true')
    glue_parser.add_argument('--load_rna', action='store_true')
    glue_parser.add_argument('--preprocess', action='store_true')
    glue_parser.add_argument('--train', action='store_true')
    glue_parser.add_argument('--offset', type=int, default=0)
    glue_parser.add_argument('--min_count', type=int, default=3)
    glue_parser.add_argument('--distal_interactions', type=int, default=None)
    glue_parser.add_argument('--filter_strata', type=float, default=None)
    glue_parser.add_argument('--n_genes', type=int, default=10000)
    glue_parser.add_argument('--gene_list', nargs='+', default=None)
    glue_parser.add_argument('--bulk_rna_sampling', action='store_true')
    glue_parser.add_argument('--bulk_n_samples', type=int, default=2000)
    glue_parser.add_argument('--bulk_n_counts', type=int, default=1000)
    glue_parser.add_argument('--bulk_hic', type=str, default=None)

    # SCORE args
    glue_parser.add_argument('--rna_file', type=str, default=None)
    glue_parser.add_argument('--gtf', type=str, default=None)
    glue_parser.add_argument('--dset', type=str, default=None)
    glue_parser.add_argument('--subname', type=str, default=None)
    glue_parser.add_argument('--scool', type=str, default=None)
    glue_parser.add_argument('--reference', type=str, default=None)
    glue_parser.add_argument('--resolution', type=str, default='100kb')
    glue_parser.add_argument('--min_depth', type=int, default=40000)
    glue_parser.add_argument('--ignore_chr_filter', action='store_true')
    glue_parser.add_argument('--ignore_filter', action='store_true')

    # extra training args
    glue_parser.add_argument('--prior', type=str, default='dcq')
    glue_parser.add_argument('--hic_weight', type=str, default=10.0)
    glue_parser.add_argument('--n_neighbors', type=int, default=15)
    glue_parser.add_argument('--lam_align', type=str, default=0.02)
    glue_parser.add_argument('--lam_graph', type=str, default=0.1)
    glue_parser.add_argument('--suffix', type=str, default='2d')
    glue_parser.add_argument('--latent_dim', type=int, default=64)
    glue_parser.add_argument('--batch_size', type=int, default=128)
    glue_parser.add_argument('--h_dim', type=int, default=128)
    glue_parser.add_argument('--h_depth', type=int, default=2)
    glue_parser.add_argument('--neg_samples', type=int, default=10)
    glue_parser.add_argument('--lr', type=float, default=2e-3)
    glue_parser.add_argument('--max_epochs', type=int, default=None)
    glue_parser.add_argument('--wandb', action='store_true')
    glue_parser.add_argument('--normalize_u', action='store_true')
    glue_parser.add_argument('--multi_strata_graph_encoder', action='store_true')
    glue_parser.add_argument('--shifted_additive', action='store_true')
    glue_parser.add_argument('--use_activation', action='store_true')
    glue_parser.add_argument('--use_attn', action='store_true')
    glue_parser.add_argument('--use_batch', type=str, default=None)
    glue_parser.add_argument('--cache_checkpoint', type=str, default=None)

    glue_args = sys.argv.index('SCORE')
    args = glue_parser.parse_args(sys.argv[1:glue_args] + sys.argv[glue_args + 1:])

    dataset_name = args.dset
    out_dir = f'{dataset_name}_data'
    if args.data_dir != '':
        out_dir = f'{args.data_dir}'
    os.makedirs(out_dir, exist_ok=True)
    prior_name = args.prior
    resolution = args.resolution
    n_distal_interactions = args.distal_interactions
    filter_strata = args.filter_strata
    min_count = args.min_count
    n_genes = args.n_genes
    hic_type = 'raw'
    if args.use_ice:
        hic_type = 'ice'
    loop_q = args.loop_q
    hic_weight = args.hic_weight
    n_neighbors = args.n_neighbors
    lam_align = args.lam_align
    lam_graph = args.lam_graph
    suffix = args.suffix
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    h_dim = args.h_dim
    h_depth = args.h_depth
    n_strata = args.n_strata
    neg_samples = args.neg_samples
    lr = args.lr
    max_epochs = args.max_epochs if args.max_epochs is not None else "AUTO"
    normalize_u = args.normalize_u
    multi_strata_graph_encoder = args.multi_strata_graph_encoder
    full_file_suffix = f"{resolution}_{hic_type}_{loop_q}_{suffix}_{n_strata}"
    graph_file_suffix = f"{prior_name}_prior_{resolution}_{hic_type}_{loop_q}_{suffix}_{n_strata}"
    use_rep = None
    min_depth = args.min_depth
    counts_per_cell = args.bulk_n_counts
    shifted_additive = args.shifted_additive
    use_activation = args.use_activation
    use_attn = args.use_attn
    use_wandb = args.wandb
    use_rna_pca = True
    use_batch = args.use_batch
    cache_checkpoint = args.cache_checkpoint
    min_confidence = 0.4

    if args.preprocess:
        preprocess_higlue(args, glue_args)

    if args.train:
        
        prior = nx.read_graphml(f"{out_dir}/graphs/{graph_file_suffix}.graphml.gz") 
        rna = ad.read_h5ad(f"{out_dir}/rna/rna_{full_file_suffix}.h5ad")
        
        hic = ad.read_h5ad(f"{out_dir}/hic/hic_{full_file_suffix}.h5ad")
        hic.obs.loc[hic.obs_names.str.startswith('alpha_'), 'celltype'] = 'Alpha'
        hic.obs.loc[hic.obs_names.str.startswith('beta_'), 'celltype'] = 'Beta'

        rna.var["highly_variable"] = rna.var["highly_variable"] & rna.var["in_hic"]
        hic.var["highly_variable"] = True
        hic = hic[hic.obs['depth'] > min_depth, :]

        # set depth as fraction of total counts
        rna.obs['depth'] = rna.layers['counts'].sum(axis=1)
        rna.obs['depth'] = rna.obs['depth'] / rna.obs['depth'].max()
        # set hic depth per batch
        for batch in hic.obs['batch'].unique():
            mask = hic.obs['batch'] == batch
            batch_hic = hic[mask, :].copy()
            batch_hic.obs['depth'] = batch_hic.layers['counts'].sum(axis=1)
            batch_hic.obs['depth'] = batch_hic.obs['depth'] / batch_hic.obs['depth'].max()
            hic.obs.loc[mask, 'depth'] = batch_hic.obs['depth'].values

        celltypes = hic.obs['celltype'].unique()
        n_clusters = len(celltypes)
        colors = list(plt.cm.tab20(np.int32(np.linspace(0, n_clusters + 0.99, n_clusters))))
        color_map = {celltype: colors[i] for i, celltype in enumerate(celltypes)}
        if 'pfc' in dataset_name:
            neurons_rna_celltypes = ['IN-PV', 'IN-SST', 'IN-VIP', 'IN-SV2C', 'Neu-mat',
                      'L2/3', 'L4', 'L5', 'L5/6', 'L6', 'L5/6-CC']
            neurons_hic_celltypes = ['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Pvalb', 'Sst', 'Vip']  

            rna_celltype_map = {'AST-FB': 'Astro',
                                'AST-PP': 'Astro',
                                'Endothelial': 'Endo',
                                'IN-PV': 'Pvalb',
                                'IN-SST': 'Sst',
                                'IN-VIP': 'Vip',
                                'IN-SV2C': 'Ndnf',
                                'L2/3': 'L2/3',
                                'L4': 'L4',
                                'L5': 'L5',
                                'L6': 'L6',
                                'L5/6': 'L6',
                                'L5/6-CC': 'L5',
                                'Microglia': 'MG',
                                'Neu-NRGN-I': 'Neu',
                                'Neu-NRGN-II': 'Neu',
                                'Neu-mat': 'Neu',
                                'OPC': 'OPC',
                                'Oligodendrocyte': 'ODC'}
            #rna = rna[rna.obs['region'] == 'PFC', :]
            #rna = rna[~rna.obs['celltype'].isin(['Neu-NRGN-I', 'Neu-NRGN-II', 'Neu-mat'])]
            rna = rna[~rna.obs['celltype'].isin(['Neu-NRGN-I', 'Neu-NRGN-II'])]
            rna = rna[rna.obs['celltype'].isin(neurons_rna_celltypes)]
            color_map = {
                "L2/3": [230, 25, 75],
                "L4": [60, 180, 75],
                "L5": [255, 225, 25],
                "L6": [0, 130, 200],
                "Ndnf": [245, 130, 49],
                "Vip": [145, 30, 180],
                "Pvalb": [70, 240, 240],
                "Sst": [240, 50, 230]}
            color_map = {celltype: [c / 255.0 for c in color] for celltype, color in color_map.items()}
            color_map['Neu'] = 'gray'
            color_map['Neu_rna'] = 'gray'

            rna.obs['celltype'] = rna.obs['celltype'].map(rna_celltype_map)
        rna_celltypes = rna.obs['celltype'].unique()
        for celltype in rna_celltypes:
            if celltype not in celltypes:
                color_map[celltype] = colors[-1]
        rna.obs['celltype'] = pd.Categorical(rna.obs['celltype'], categories=rna_celltypes, ordered=True)
        hic.obs['celltype'] = pd.Categorical(hic.obs['celltype'], categories=celltypes, ordered=True)

        if use_wandb:
            import wandb
            wandb.init(project=f'GLUE-{dataset_name}-hic-2d-prior', 
                    sync_tensorboard=True, 
                    config={'prior': prior_name, 
                            'res': resolution,
                            'hic_type': hic_type,
                            'loop_q': loop_q,
                            'suffix': suffix, 
                            'n_distal_interactions': n_distal_interactions,
                            'counts_per_cell_rna': counts_per_cell,
                            'filter_strata': filter_strata,
                            'min_count': min_count,
                            'n_genes': n_genes,
                            'rna_pca': use_rna_pca,
                            'latent_dim': latent_dim,
                            'h_dim': h_dim,
                            'h_depth': h_depth,
                            'n_strata': n_strata,
                            'batch_size': batch_size,
                            'neg_samples': neg_samples,
                            'normalize_u': normalize_u,
                            'shifted_additive': shifted_additive,
                            'use_activation': use_activation,
                            'use_attn': use_attn,
                            'multi_strata_graph_encoder': multi_strata_graph_encoder,
                            'lr': lr,
                            'hic_weight': hic_weight,
                            'n_neighbors': n_neighbors,
                            'lam_align': lam_align,
                            'lam_graph': lam_graph})

        scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_layer="counts", use_rep="X_pca" if use_rna_pca else None,
                                        use_cell_type=None, use_batch=use_batch, use_depth="depth")
        scglue.models.configure_dataset(hic, "HiCZINB", use_highly_variable=True, use_layer="counts", 
                                        use_depth="depth", use_batch="batch")

        print(f"Total nodes in prior: {len(prior.nodes)}")

        glue = scglue.models.fit_SCGLUE(
            {"rna": rna, "hic": hic}, prior,
            log_wandb=use_wandb,
            init_kws={"latent_dim": latent_dim, 
                    "use_multi_strata_graph_encoder": multi_strata_graph_encoder, 
                    "shifted_additive": shifted_additive,
                    "use_activation": use_activation,
                    "use_attn": use_attn,
                    "h_dim": h_dim,
                    "h_depth": h_depth,
                    "n_strata": n_strata},
            compile_kws={"lam_align": lam_align, 
                        "lam_graph": lam_graph,
                        "normalize_u": normalize_u,
                        "lr": lr,
                        "modality_weight": {'rna': 1.0, 'hic': hic_weight}},
            balance_kws={"resolution": 1.0},
            fit_kws={"directory": "glue", 
                    "neg_samples": neg_samples,
                    "val_split": 0.05,
                    "data_batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "wait_n_lrs": 2}
        )

        glue.save(f"{out_dir}/glue_hic_{prior_name}_prior_{resolution}_{n_genes}_{n_strata}.dill")
        if cache_checkpoint:
            n_checkpoints = len(os.listdir(cache_checkpoint))
            glue.save(f"{cache_checkpoint}/glue_hic_{prior_name}_prior_{resolution}_{n_checkpoints}.dill")
        # embed and visualize
        rna.obsm["X_glue"] = glue.encode_data("rna", rna)
        hic.obsm["X_glue"] = glue.encode_data("hic", hic)

        rna.obs['domain'] = 'rna'
        hic.obs['domain'] = 'hic'

        rna.obs['old_celltype'] = rna.obs['celltype']
        hic.obs['old_celltype'] = hic.obs['celltype']
        # run leiden clustering to identify clusters
        sc.pp.neighbors(hic, use_rep="X_glue", metric="cosine")
        sc.tl.leiden(hic)

        sc.pp.neighbors(rna, use_rep="X_glue", metric="cosine")
        sc.tl.leiden(rna)

        # transfer labels to predict celltypes
        scglue.data.transfer_labels(rna, hic, "celltype", use_rep="X_glue", n_neighbors=n_neighbors)
        try:
            # map celltypes to integers
            celltypes = hic.obs['old_celltype'].unique()
            celltype_map = {c: i for i, c in enumerate(celltypes)}
            hic.obs['old_celltype_int'] = hic.obs['old_celltype'].map(celltype_map)
            hic.obs['celltype_int'] = hic.obs['celltype'].map(celltype_map)
            hic.obs['celltype_int'].fillna(len(celltypes), inplace=True)
            # compute full accuracy
            accuracy = accuracy_score(hic.obs['old_celltype_int'], hic.obs['celltype_int'])
            ari = adjusted_rand_score(hic.obs['old_celltype_int'], hic.obs['celltype_int'])
            ari_leiden = adjusted_rand_score(hic.obs['old_celltype_int'], hic.obs['leiden'])
            if use_wandb:
                wandb.log({"accuracy": accuracy, 
                        "ari": ari_leiden,
                        "ari_label_transfer": ari})

            # also do the same for RNA
            rna_celltypes = rna.obs['celltype'].unique()
            print(rna_celltypes)
            rna_celltype_map = {c: i for i, c in enumerate(rna_celltypes)}
            print(rna_celltype_map)
            rna.obs['old_celltype_int'] = rna.obs['old_celltype'].map(rna_celltype_map)
            #rna.obs['celltype_int'] = rna.obs['leiden'].map(rna_celltype_map)
            #rna.obs['celltype_int'].fillna(len(rna_celltypes), inplace=True)
            rna_accuracy = accuracy_score(rna.obs['old_celltype_int'], rna.obs['leiden'])
            rna_ari = adjusted_rand_score(rna.obs['old_celltype_int'], rna.obs['leiden'])
            if use_wandb:
                wandb.log({"rna_accuracy": rna_accuracy, 
                        "rna_ari": rna_ari})

            # remove low confidence cells
            hic = hic[hic.obs['celltype_confidence'] > min_confidence, :].copy()
            #hic = hic[hic.obs['depth'] > min_depth, :]

            # compute filtered accuracy
            accuracy = accuracy_score(hic.obs['old_celltype_int'], hic.obs['celltype_int'])
            ari = adjusted_rand_score(hic.obs['old_celltype_int'], hic.obs['celltype_int'])
            ari_leiden = adjusted_rand_score(hic.obs['old_celltype_int'], hic.obs['leiden'])
            if use_wandb:
                wandb.log({"accuracy_filtered": accuracy, 
                        "ari_filtered": ari_leiden,
                        "ari_filtered_label_transfer": ari})
        except Exception as e:
            print(e)
            pass

        combined = ad.concat([rna, hic])

        sc.pp.neighbors(hic, use_rep="X_glue", metric="cosine", n_neighbors=n_neighbors)
        sc.tl.umap(hic)
        fig = sc.pl.umap(hic, color=["old_celltype", "celltype", "celltype_confidence", "depth", "batch"], palette=color_map,wspace=0.45, return_fig=True)
        fig.savefig('glue_umap.png')
        plt.close()
        if use_wandb:
            wandb.log({"hic_umap": wandb.Image('glue_umap.png')})

        sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine", n_neighbors=n_neighbors)
        sc.tl.umap(combined)
        fig = sc.pl.umap(combined, color=["domain"], wspace=0.45, return_fig=True)
        plt.tight_layout()
        fig.savefig('glue_joint_umap.png')
        plt.close()
        if use_wandb:
            wandb.log({"joint_umap": wandb.Image('glue_joint_umap.png')})

        fig = sc.pl.umap(combined[combined.obs["domain"] == "hic"], color=["old_celltype", "celltype"], palette=color_map, wspace=0.45, return_fig=True)
        fig.savefig('glue_joint_umap_hic.png')
        plt.close()
        if use_wandb:
            wandb.log({"joint_umap_hic": wandb.Image('glue_joint_umap_hic.png')})

        fig = sc.pl.umap(combined[combined.obs["domain"] == "rna"], color=["celltype"], palette=color_map, wspace=0.45, return_fig=True)
        fig.savefig('glue_joint_umap_rna.png')
        plt.close()
        if use_wandb:
            wandb.log({"joint_umap_rna": wandb.Image('glue_joint_umap_rna.png')})

        rna_mask = combined.obs['domain'] == 'rna'
        combined.obs['old_celltype'] = combined.obs['old_celltype'].astype(str)
        combined.obs.loc[rna_mask, 'old_celltype'] += '_rna'
        
        rna_color_map = {celltype + '_rna': colors[i] for i, celltype in enumerate(celltypes)}
        # add any missing celltypes from RNA
        rna_celltypes = rna.obs['celltype'].unique()
        for celltype in rna_celltypes:
            if celltype not in celltypes:
                rna_color_map[celltype + '_rna'] = colors[-1]
        color_map = {**color_map, **rna_color_map}
        color_map['Other'] = 'gray'
        fig = sc.pl.umap(combined, color=["old_celltype"], groups=celltypes, palette=color_map, size=100, wspace=0.45, return_fig=True)
        fig.savefig('glue_joint_umap_separate.png')
        plt.close()
        if use_wandb:
            wandb.log({"joint_umap_separate": wandb.Image('glue_joint_umap_separate.png')})

        # save combined data
        os.makedirs(f"{out_dir}/combined_embedding", exist_ok=True)
        combined.write(f"{out_dir}/combined_embedding/combined_{full_file_suffix}.h5ad", compression="gzip")

        if 'islet' in dataset_name:
            sorted_hic = hic[hic.obs_names.str.startswith('alpha_') | hic.obs_names.str.startswith('beta_')]
            sorted_rna = rna[rna.obs['celltype'].isin(['Alpha', 'Beta'])]
            sorted_hic.obs['sorted_celltype'] = sorted_hic.obs['celltype']
            celltypes = sorted(sorted_hic.obs['celltype'].unique())
            celltype_map = {c: i for i, c in enumerate(celltypes)}
            sorted_hic.obs['celltype_int'] = sorted_hic.obs['celltype'].map(celltype_map)
            scglue.data.transfer_labels(sorted_rna, sorted_hic, "celltype", use_rep="X_glue", n_neighbors=5, key_added="pred_celltype_sorted")
            sorted_hic.obs['pred_celltype_int'] = sorted_hic.obs['pred_celltype_sorted'].map(celltype_map)
            # measure accuracy
            val_accuracy = accuracy_score(sorted_hic.obs['celltype_int'], sorted_hic.obs['pred_celltype_int'])
            val_ari = adjusted_rand_score(sorted_hic.obs['celltype_int'], sorted_hic.obs['pred_celltype_int'])
            if use_wandb:
                wandb.log({"val_accuracy": val_accuracy, "val_ari": val_ari})

            confident_filtered_hic = sorted_hic[sorted_hic.obs['celltype_confidence'] > min_confidence, :].copy()
            confident_filtered_hic = confident_filtered_hic[confident_filtered_hic.obs['depth'] > min_depth, :]
            # measure accuracy
            val_accuracy = accuracy_score(confident_filtered_hic.obs['celltype_int'], confident_filtered_hic.obs['pred_celltype_int'])
            val_ari = adjusted_rand_score(confident_filtered_hic.obs['celltype_int'], confident_filtered_hic.obs['pred_celltype_int'])
            if use_wandb:
                wandb.log({"val_accuracy_filtered": val_accuracy, "val_ari_filtered": val_ari})

        if use_wandb:
            wandb.finish()