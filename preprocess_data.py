import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import cooler
from matplotlib import rcParams
from cooler._logging import set_verbosity_level
import os
import sys
import itertools
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool

from tqdm import tqdm
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, csr_matrix
from networkx.algorithms.bipartite import biadjacency_matrix
from score.sc_args import parse_args
from score.utils.utils import anchor_to_locus, anchor_list_to_dict, sorted_nicely
from score.utils.matrix_ops import VC_SQRT_norm

def OE_norm(mat, max_strata=100):
    new_mat = mat.copy() / np.max(mat)  # unchanged values guaranteed to be <=1
    #averages = np.array([np.mean(mat[i:, :len(mat) - i]) for i in range(min(max_strata, len(mat)))])
    averages = np.array([np.mean(np.diagonal(mat, offset=i)) for i in range(min(max_strata, len(mat)))])
    averages = np.where(averages == 0, 1, averages)
    # for i in range(len(mat)):
    #     for j in range(len(mat)):
    #         d = abs(i - j)
    #         if d < max_strata:
    #             new_mat[i, j] = mat[i, j] / averages[d]
    # # vectorized version
    for i in range(min(max_strata, len(mat))):
        new_mat[i:, :len(mat) - i] = mat[i:, :len(mat) - i] / averages[i]
    return new_mat

def get_processed_matrix(dataset, cell, cell_i, preprocessing, chr_only=None):
    c = cooler.Cooler(f"{dataset.scool_file}::/cells/{cell}")
    if chr_only is None:
        chr_list = list(pd.unique(dataset.anchor_list['chr'])) if chr_only is None else [chr_only]
        loops = c.pixels()[:]
        chr_mats = []
        for chr_name in chr_list:
            if chr_only is not None and chr_name != chr_only:
                continue
            chr_anchors = dataset.anchor_list.loc[dataset.anchor_list['chr'] == chr_name]
            chr_anchors.reset_index(drop=True, inplace=True)
            chr_anchor_dict = anchor_list_to_dict(chr_anchors['anchor'].values)
            chr_contacts = loops.loc[loops['a1'].isin(chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])].copy()
            if len(chr_contacts) > 0:
                chr_contacts['chr1'] = chr_name
                chr_contacts['chr2'] = chr_name

                rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                    chr_contacts['a1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                    chr_contacts['a2'].values)  # convert anchor names to column indices
                matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
                            shape=(len(chr_anchors), len(chr_anchors)))
                mat = matrix.toarray()
            
            else:
                mat = np.zeros((len(chr_anchors), len(chr_anchors)))
            chr_mats.append(mat)
        mat = csr_matrix(block_diag(*chr_mats))
    else:
        mat = c.matrix(sparse=True).fetch(chr_only)
    return cell_i, cell, mat


def get_flattened_matrices(dataset, n_strata, preprocessing=None, agg_fn=None, chr_only=None, offset=0, n_proc=8):
    mats = {}
    results = []
    with Pool(n_proc) as p:
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            results.append(p.apply_async(get_processed_matrix, args=(dataset, cell, cell_i, preprocessing, chr_only)))
        for res in tqdm(results):
            cell_i, cell, mat = res.get(timeout=1000)
            if chr_only is not None:
                new_mat = []
                for i in range(n_strata):
                    new_strata = list(mat.diagonal(k=i + offset))
                    if len(new_strata) < mat.shape[0]:
                        new_strata += [0] * (mat.shape[0] - len(new_strata))
                    new_mat.append(new_strata)
                new_mat = np.concatenate(new_mat)
                mats[cell] = new_mat
            else:
                mats[cell] = mat

    full_mats = []
    for cell_i, cell in enumerate(sorted(dataset.cell_list)): 
        full_mats.append(mats[cell])

    if agg_fn is not None:  # aggregating to 1D
        print('Aggregating to 1D')
        strata_mask = np.zeros_like(full_mats[0].toarray())
        for k in range(n_strata):
            strata_mask += np.eye(strata_mask.shape[0], k=k, dtype=full_mats[0].dtype)
        mat = []
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            tmp_mat = full_mats[cell_i].toarray()
            tmp_mat[strata_mask == 0] = 0
            mat.append(agg_fn(tmp_mat, axis=0))
    else:  # unraveling 2D strata to 1D vector
        print('Unraveling 2D strata to 1D vector')
        mat = []
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            if chr_only is not None:
                counts = full_mats[cell_i]
                mat.append(counts)
            else:
                counts = []
                chr_size = full_mats[cell_i].shape[0]
                for k in range(n_strata):
                    new_strata = list(full_mats[cell_i].diagonal(k=k + offset))
                    print(len(new_strata), chr_size)
                    if len(new_strata) < chr_size:
                        new_strata += [0] * (chr_size - len(new_strata))
                    counts += new_strata
                    
                mat.append(counts)
    x = np.array(mat)
    return x



def preprocess_higlue(args, glue_args):
    n_strata = args.n_strata
    resolution = args.resolution
    gtf_file = args.gtf
    rna_file = args.rna_file
    atac_file = args.atac_file
    methyl_file = args.methyl_file
    loop_q = args.loop_q
    load_rna = args.load_rna
    load_hic = False
    use_toploops = False 
    use_ice = args.use_ice
    use_dist_norm = args.use_dist_norm
    use_trans = args.use_trans
    viz_rna = args.viz_rna
    use_raw_pseudobulk = True
    loops_offset = args.offset
    min_count = args.min_count
    n_distal_interactions = args.distal_interactions
    filter_strata = args.filter_strata
    exclusive_strata = args.exclusive_strata
    use_xy = args.use_xy
    n_genes = args.n_genes
    gene_list = args.gene_list
    bulk_hic = args.bulk_hic
    if not use_toploops:
        loop_q = float(loop_q)
    else:
        if loop_q.endswith('k'):
            n_loops = int(loop_q[:-1]) * 1000
    dataset_name = args.dset
    out_dir = f'data/{dataset_name}_data'
    plot_dir = f'plots/{dataset_name}_plots'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'rna'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'graphs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'hic'), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print('Parsing SCORE args...')
    parser = argparse.ArgumentParser()
    sys.argv = sys.argv[glue_args - 1:]
    args, x, y, depths, batches, dataset, valid_dataset = parse_args(parser)

    if bulk_hic is not None:
        bulk = cooler.Cooler(bulk_hic)
    else:
        cool_files = cooler.fileops.list_scool_cells(dataset.scool_file)
        cool_files = [f"{dataset.scool_file}::{cell}" for cell in cool_files]
        os.makedirs('data/scools', exist_ok=True)
        out_cool_file = f"data/scools/{dataset_name}_{resolution}_bulk.cool"

        try:
            bulk = cooler.Cooler(out_cool_file)
            if use_ice:
                if 'weight' not in bulk.bins().columns:
                    cooler.balance_cooler(bulk, cis_only=True, store=True)
        except Exception as e:
            print('Could not find bulk cooler file, generating from psuedobulk...')
            set_verbosity_level(1)
            cooler.merge_coolers(out_cool_file, cool_files, mergebuf=40000000)
            bulk = cooler.Cooler(out_cool_file)
            if use_ice:
                cooler.balance_cooler(bulk, cis_only=False, store=True)
                bulk = cooler.Cooler(out_cool_file)
    # plot example heatmap
    mat = bulk.matrix(balance=use_ice).fetch('chr10')
    midpoint = mat.shape[0] // 2
    mat = mat[midpoint - n_strata:midpoint + n_strata, midpoint - n_strata:midpoint + n_strata]
    plt.imshow(mat, cmap='Reds', norm=LogNorm())
    plt.colorbar()
    plt.savefig(f'{plot_dir}/hic_example.png')
    plt.close()

    frags = bulk.bins()[:]
    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)
    frags['name'] = frags.index.astype(str)
    loops = bulk.pixels(join=False)[:]
    chr_map = frags['chrom'].to_dict()
    start_map = frags['chromStart'].to_dict()
    end_map = frags['chromEnd'].to_dict()
    loops['chr1'] = loops['bin1_id'].map(chr_map)   
    loops['chr2'] = loops['bin2_id'].map(chr_map)

    if use_trans:
        # use total of same loop_q trans loops by transforming based on number of chroms
        trans_loop_q = 1 - (1 - loop_q) / (len(bulk.chromnames) * (len(bulk.chromnames) - 1) / 2)
        print(f"Trans loop quantile:{trans_loop_q:.4f}")
        trans = []
        used_chrs = []
        print('Loading trans chromosomal interactions...')
        for chrom1 in tqdm(bulk.chromnames):
            chr_trans_pixels = []
            for chrom2 in bulk.chromnames:
                if chrom1 == chrom2:
                    continue
                if (chrom1, chrom2) in used_chrs or (chrom2, chrom1) in used_chrs:
                    continue
                used_chrs.append((chrom1, chrom2))
                #trans_pixels = bulk.pixels(join=False).fetch(chrom1, chrom2)
                trans_pixels = loops[(loops['chr1'] == chrom1) & (loops['chr2'] == chrom2)].copy()
                trans_pixels.drop(columns=['chr1', 'chr2'], inplace=True)
                chr_trans_pixels.append(trans_pixels)
            if len(chr_trans_pixels) > 0:
                chr_trans_pixels = pd.concat(chr_trans_pixels).reset_index(drop=True)
                chr_trans_pixels['rank'] = chr_trans_pixels['count'].rank(pct=True)
                try:
                    loop_cutoff = np.quantile(chr_trans_pixels['rank'].values, q=trans_loop_q)
                    chr_trans_pixels = chr_trans_pixels.loc[chr_trans_pixels['rank'] >= loop_cutoff].copy()
                    chr_trans_pixels['rank'] = chr_trans_pixels['rank'].rank(pct=True)
                    #chr_trans_pixels['rank'] = chr_trans_pixels['rank'] * 0.5 + 0.5
                    trans.append(chr_trans_pixels)
                except IndexError:  # no reads in chrom (e.g no chrY)
                    pass  
        trans = pd.concat(trans).reset_index(drop=True)
        print('Trans interactions:')
        print(trans)
    if exclusive_strata:
        # remove interactions beyond n_strata
        loops['strata'] = abs(loops['bin1_id'] - loops['bin2_id'])
        loops = loops[loops['strata'] <= n_strata].copy()
        loops.drop(columns=['strata'], inplace=True)

    if use_ice:
        weight_map = frags['weight'].to_dict()
        loops['weight1'] = loops['bin1_id'].map(weight_map)
        loops['weight2'] = loops['bin2_id'].map(weight_map)
        loops['oe'] = loops['count'] * loops['weight1'] * loops['weight2']
        loops['rank'] = loops['oe'].rank(pct=True)
    else:
        loops['rank'] = loops['count'].rank(pct=True)
    loops.dropna(inplace=True)

    if use_dist_norm:
        # extract each dense matrix and compute OE normalization
        print('OE normalizing Hi-C data...')
        oe_loops = []
        for chr_name in tqdm(sorted_nicely(bulk.chromnames)):
            if not use_xy and ('x' in chr_name.lower() or 'y' in chr_name.lower()):
                continue
            chr_loops = loops[(loops['chr1'] == chr_name) & (loops['chr2'] == chr_name)].copy()
            chr_anchors = frags[frags['chrom'] == chr_name].copy()
            chr_offset = chr_anchors.index[0]
            chr_anchors.reset_index(drop=True, inplace=True)
            
            # convert to dense matrix
            if use_ice:
                val_col = 'oe'
            else:
                val_col = 'count'
            chr_mat = csr_matrix((chr_loops[val_col].values, 
                                  (chr_loops['bin1_id'].values - chr_offset, chr_loops['bin2_id'].values - chr_offset)), 
                                  shape=(len(chr_anchors), len(chr_anchors)))
            # chr_mat = bulk.matrix(balance=use_ice).fetch(chr_name)
            chr_mat = OE_norm(VC_SQRT_norm(chr_mat.toarray()))
            chr_mat = csr_matrix(chr_mat)
            oe_values = chr_mat.data
            chr_loops['oe'] = oe_values
            oe_loops.append(chr_loops)
            if chr_name == 'chr10':  # visualize first matrix
                mat = chr_mat.toarray()
                mat = mat + mat.T - np.diag(np.diag(mat))
                midpoint = mat.shape[0] // 2
                mat = mat[midpoint - n_strata:midpoint + n_strata, midpoint - n_strata:midpoint + n_strata]
                plt.imshow(mat, cmap='Reds', norm=LogNorm())
                plt.colorbar()
                plt.savefig(f'{plot_dir}/hic_example_oe.png')
                plt.close()
        loops = pd.concat(oe_loops).reset_index(drop=True)
        loops['rank'] = loops['oe'].rank(pct=True)
        print(loops)
    
    loop_dfs = []
    print('Filtering top loops in each chromosome...')
    for chr_name in tqdm(sorted_nicely(bulk.chromnames)):
        if not use_xy and ('x' in chr_name.lower() or 'y' in chr_name.lower()):
            continue
        chr_loops = loops[(loops['chr1'] == chr_name) & (loops['chr2'] == chr_name)].copy()
        chr_loops['rank'] = chr_loops['oe' if use_ice or use_dist_norm else 'count'].rank(pct=True)
        loop_cutoff = np.quantile(chr_loops['rank'].values, q=loop_q)
        chr_loops = chr_loops.loc[chr_loops['rank'] >= loop_cutoff].copy()
        chr_loops['rank'] = chr_loops['rank'].rank(pct=True)
        chr_loops.reset_index(drop=True, inplace=True)
        chr_loops.drop(columns=['chr1', 'chr2'], inplace=True)
        if use_ice:
            chr_loops.drop(columns=['weight1', 'weight2', 'oe'], inplace=True)
        loop_dfs.append(chr_loops)
    loops = pd.concat(loop_dfs).reset_index(drop=True)
    
    if use_trans:
        loops = pd.concat([loops, trans]).reset_index(drop=True)

    print('Final loops:')
    print(loops)
    if not use_xy:
        frags = frags[~frags['chrom'].str.lower().str.contains('x|y')].reset_index(drop=True)
    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)

    base_rna_filename = 'rna_base_2d.h5ad'
    if  base_rna_filename not in os.listdir(os.path.join(out_dir, 'rna')) or not load_rna:
        rna = ad.read_h5ad(rna_file)
        try:
            rna.X = rna.layers["counts"]
        except Exception as e:
            pass
        
        if 'batch' not in rna.obs.columns:
            rna.obs['batch'] = 0
        rna.layers["counts"] = rna.X.copy()
        keep_columns = scglue.genomics.Bed.COLUMNS
        for col in keep_columns:
            try:
                rna.var.drop(columns=[col], inplace=True)
            except Exception as e:
                print(e)
                pass
        try:
            scglue.data.get_gene_annotation(
                rna, gtf=gtf_file,
                gtf_by="gene_name"
            )
        except KeyError:
            scglue.data.get_gene_annotation(
                rna, gtf=gtf_file,
                gtf_by="gene_symbol"
            )
            
        
        rna.var['chromStart'] = rna.var['chromStart'].fillna(0).astype(int)
        rna.var['chromEnd'] = rna.var['chromEnd'].fillna(0).astype(int)
        rna.var['strand'] = rna.var['strand'].fillna('+')
        rna = rna[:, rna.var['chrom'].notna()].copy()
        try:
            if not rna.var['chrom'].iloc[0].startswith('chr'):
                rna.var['chrom'] = 'chr' + rna.var['chrom']
        except AttributeError:
            print(rna.var['chrom'].iloc[0])
            print(rna.var['chrom'])

        drop_cols = []
        for col in rna.var.columns:
            if col not in keep_columns:
                drop_cols.append(col)
        rna.var.drop(columns=drop_cols, inplace=True)
        rna = rna[:, rna.var['chrom'].notna()].copy()
        if not use_xy:
            rna = rna[:, ~rna.var['chrom'].str.lower().str.contains('x|y')].copy()
        genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
        rna.write(f"{out_dir}/rna/{base_rna_filename}", compression="gzip")
    else:
        rna = ad.read_h5ad(f"{out_dir}/rna/{base_rna_filename}")
    
    sc.pp.filter_genes(rna, min_counts=1)
    print('Embedding RNA...')
    print('Highly variable genes...')
    
    sc.pp.highly_variable_genes(rna, n_top_genes=n_genes, flavor="seurat_v3", span=1)
    if gene_list is not None:
        print('Adding user provided gene list to highly variable genes...')
        for gene in gene_list:
            if gene in rna.var_names:
                rna.var.loc[gene, 'highly_variable'] = True
        
    # # #sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print("Normalize")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    print("PCA")
    sc.tl.pca(rna, n_comps=200, svd_solver="auto")
    if viz_rna:
        sc.pp.neighbors(rna, n_pcs=200, metric="cosine")
        sc.tl.umap(rna)

        fig = sc.pl.umap(rna, color=["celltype", "batch"], return_fig=True, wspace=0.6)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/rna_umap.png")
        plt.close()


    base_hic_filename = f"hic_base_{resolution}_2d.h5ad"
    base_hic_path = f"{out_dir}/hic/{base_hic_filename}"
    if base_hic_filename not in os.listdir(os.path.join(out_dir, 'hic')) or not load_hic:
        print('Loading scHi-C sparse matrices...')
        mats = dataset.get_sparse_matrices()
        mat = []
        strata_adatas = []
        total_interactions = 0
        print(f'Processing strata...')
        for k in range(abs(loops_offset), n_strata + abs(loops_offset)):
            strata_mat = []
            for cell_i, cell in enumerate(sorted(dataset.cell_list)):
                new_strata = list(mats[cell_i].diagonal(k=k))
                if len(new_strata) < len(dataset.anchor_list):
                    new_strata += [0] * (len(dataset.anchor_list) - len(new_strata))
                if resolution in ['10kb', '20kb', '50kb']:
                    strata_mat.append(np.uint8(new_strata))
                else:
                    strata_mat.append(new_strata)
            # create per-strata anndata
            strata_hic = ad.AnnData(np.array(strata_mat), dtype=np.uint8 if resolution in ['10kb', '20kb', '50kb'] else np.int32)
            strata_hic.obs_names = sorted(dataset.cell_list)
            strata_hic.obs_names = strata_hic.obs_names.map(lambda s: s.replace(f'.{dataset.res_name}', ''))
            genomic_pos = dataset.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
            # if not use_xy:
            #     genomic_pos = genomic_pos[~genomic_pos.str.lower().str.contains('x|y')].reset_index(drop=True)
            if k - loops_offset == 0:
                strata_hic.var_names = genomic_pos
            else:
                print(strata_hic.var_names.shape, genomic_pos.shape)
                if strata_hic.var_names.shape != genomic_pos.shape:
                    strata_hic.var_names = genomic_pos.iloc[:-(k - loops_offset)] + f'-{k - loops_offset}'
                else:
                    strata_hic.var_names = genomic_pos + f'-{k - loops_offset}'
            strata_hic.var['root'] = strata_hic.var_names.map(lambda s: s.rsplit('-', 1)[0] if (s[-2] == '-' or s[-3] == '-') else s)
            split = strata_hic.var['root'].str.split(r"[:-]")
            strata_hic.var["chrom"] = split.map(lambda x: x[0])
            strata_hic.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
            strata_hic.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
            if not use_xy:
                strata_hic = strata_hic[:, ~strata_hic.var['chrom'].str.lower().str.contains('x|y')].copy()
            next_strata_map = frags.copy()
            next_strata_map['name'] = next_strata_map.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", axis=1)
            roots = next_strata_map['name'].values
            next_vars = np.roll(roots, shift=-k)
            next_vars[-k:] = roots[-k:]
            next_strata_map['target'] = next_vars
            next_strata_map = next_strata_map.set_index('name').to_dict()['target']

            strata_hic.var['target'] = strata_hic.var['root'].map(next_strata_map)
            split = strata_hic.var['target'].str.split(r"[:-]")
            strata_hic.var["target_chrom"] = split.map(lambda x: x[0])
            strata_hic.var["target_chromStart"] = split.map(lambda x: x[1]).astype(int)
            strata_hic.var["target_chromEnd"] = split.map(lambda x: x[2]).astype(int)

            if k - loops_offset > 0:  # keep all features for the first strata as roots for later, even if they have no interactions
                sc.pp.filter_genes(strata_hic, min_counts=min_count)
                sc.pp.filter_genes(strata_hic, min_cells=min_count)
                if filter_strata:
                    # find highly variable features
                    sc.pp.highly_variable_genes(strata_hic, n_top_genes=int(strata_hic.shape[1] * filter_strata), flavor="seurat_v3", span=1)
                    top_loop_mask = strata_hic.var['highly_variable'].values
                    strata_hic = strata_hic[:, top_loop_mask].copy()
            strata_hic.var.drop(columns=['root'], inplace=True)
            strata_adatas.append(strata_hic)
            total_interactions += strata_hic.shape[1]
            print(f'{k} - Total interactions: {total_interactions:,} ({strata_hic.shape[1]:,} from strata {k})')
        
        hic = ad.concat(strata_adatas, axis=1, join='inner')
        hic.layers["counts"] = hic.X.copy()
        hic.obs['celltype'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'cluster'] for cell in hic.obs_names])
        hic.obs['depth'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'depth'] for cell in hic.obs_names])
        hic.obs['batch'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'batch'] for cell in hic.obs_names])

        hic.obs['dataset'] = 'train'
        hic.write(base_hic_path, compression="gzip")
    else:
        hic = ad.read_h5ad(base_hic_path)
    print('Analyzing Hi-C data...')
    top_loop_mask = np.ones(hic.shape[1], dtype=bool)
    if n_distal_interactions is not None:
        if hic.shape[1] > n_distal_interactions:
            print('Identifying highly variable interactions...')
            sc.pp.highly_variable_genes(hic, n_top_genes=n_distal_interactions, flavor="seurat_v3", span=1)
            top_loop_mask = hic.var['highly_variable'].values
        
    print('Filtering distal interactions...')
    gene_mask_count, _ = sc.pp.filter_genes(hic, min_counts=min_count, inplace=False)
    gene_mask_cells, _ = sc.pp.filter_genes(hic, min_cells=min_count, inplace=False)
    gene_mask = gene_mask_count & gene_mask_cells & top_loop_mask
    non_distal_mask = np.array(hic.var_names.map(lambda s: s[-2] != '-' and s[-3] != '-').values, dtype=bool)
    gene_mask = gene_mask | non_distal_mask # create non distal mask
    # then figure out which root non distal features need to be kept
    keep_distal_mask = hic.var_names.map(lambda s: s[-2] == '-' or s[-3] == '-')
    distal_source = hic.var_names[keep_distal_mask].map(lambda s: s.rsplit('-', 1)[0]).values
    hic = hic[:, gene_mask].copy()  # first filter out all distal features
    # then filter out the root non distal features that are not needed
    keep_non_distal_mask = hic.var_names.isin(distal_source)
    # now filter out the non distal features keeping the ones that are needed
    gene_mask_count, _ = sc.pp.filter_genes(hic, min_counts=min_count, inplace=False)
    gene_mask_cells, _ = sc.pp.filter_genes(hic, min_cells=min_count, inplace=False)
    gene_mask = gene_mask_count & gene_mask_cells
    gene_mask = gene_mask | keep_non_distal_mask
    hic = hic[:, gene_mask].copy()
    print(hic)
    #sc.pp.normalize_per_cell(hic, counts_per_cell_after=1e5)
    sc.tl.pca(hic, n_comps=min(100, hic.shape[0]), svd_solver="auto")
    sc.pp.neighbors(hic, n_pcs=min(100, hic.shape[0]), metric="cosine")
    # scglue.data.lsi(hic, n_components=100, n_iter=50, n_oversamples=20)
    # sc.pp.neighbors(hic, use_rep="X_lsi", metric="cosine")
    sc.tl.umap(hic)
    fig = sc.pl.umap(hic, color=["celltype", "batch"], return_fig=True)
    fig.savefig(f'{plot_dir}/hic_umap_{resolution}.png')
    plt.close() 

    if atac_file is not None:
        atac = ad.read_h5ad(atac_file)
        if 'batch' not in atac.obs.columns:
            atac.obs['batch'] = 0
        atac.layers["counts"] = atac.X.copy()
        atac.var['chrom'] = atac.var_names.map(lambda s: s.split(':')[0])
        atac.var['chromStart'] = atac.var_names.map(lambda s: s.split(':')[1].split('-')[0]).astype(int)
        atac.var['chromEnd'] = atac.var_names.map(lambda s: s.split(':')[1].split('-')[1]).astype(int)
        atac.var['name'] = atac.var_names
        atac = atac[:, atac.var['chrom'].notna()].copy()
        # only accept chroms 1-22, X and Y
        atac = atac[:, atac.var['chrom'].isin([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])].copy()
        atac_peaks = scglue.genomics.Bed(atac.var.assign(name=atac.var_names))
        sc.pp.filter_genes(atac, min_counts=2)
        # use same number of features as in Hi-C data
        sc.pp.highly_variable_genes(atac, n_top_genes=hic.shape[1], flavor="seurat_v3", span=1)
        sc.pp.normalize_total(atac)
        sc.pp.log1p(atac)
        sc.pp.scale(atac)
        sc.tl.pca(atac, n_comps=200, svd_solver="auto")
        sc.pp.neighbors(atac, n_pcs=200, metric="cosine")
        sc.tl.umap(atac)
        fig = sc.pl.umap(atac, color=["celltype", "batch"], return_fig=True, wspace=0.6)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/atac_umap.png")
        plt.close()
    if methyl_file is not None:
        methyl = ad.read_h5ad(methyl_file)
        if 'batch' not in methyl.obs.columns:
            methyl.obs['batch'] = 0
        methyl.layers["counts"] = methyl.X.copy()
        methyl.var['name'] = methyl.var_names
        methyl.var['name'] = methyl.var['name'].apply(lambda s: s.replace('_mCH', '').replace('_mCG', ''))
        keep_columns = scglue.genomics.Bed.COLUMNS
        for col in keep_columns:
            try:
                rna.var.drop(columns=[col], inplace=True)
            except Exception as e:
                print(e)
                pass
        try:
            scglue.data.get_gene_annotation(
                rna, gtf=gtf_file,
                gtf_by="gene_name"
            )
        except KeyError:
            scglue.data.get_gene_annotation(
                rna, gtf=gtf_file,
                gtf_by="gene_symbol"
            )
        methyl.var['chromStart'] = methyl.var['chromStart'].fillna(0).astype(int)
        methyl.var['chromEnd'] = methyl.var['chromEnd'].fillna(0).astype(int)
        methyl.var['strand'] = methyl.var['strand'].fillna('+')
        methyl = methyl[:, methyl.var['chrom'].notna()].copy()
        # only accept chroms 1-22, X and Y
        methyl = methyl[:, methyl.var['chrom'].isin([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])].copy()
        # add _mCH suffix to var names to distinguish from RNA genes
        #methyl.var_names = methyl.var_names.map(lambda s: s + '_mCH')
        # only keep methylation for genes in RNA data
        methyl = methyl[:, methyl.var['name'].isin(rna.var_names)].copy()
        methyl_genes = scglue.genomics.Bed(methyl.var.assign(name=methyl.var_names))
        sc.pp.filter_genes(methyl, min_counts=2)
        # use same number of genes as in RNA data
        sc.pp.highly_variable_genes(methyl, n_top_genes=n_genes, flavor="seurat_v3", span=1)
        sc.pp.normalize_total(methyl)
        sc.pp.log1p(methyl)
        sc.pp.scale(methyl)
        sc.tl.pca(methyl, n_comps=200, svd_solver="auto")
        sc.pp.neighbors(methyl, n_pcs=200, metric="cosine")
        sc.tl.umap(methyl)
        fig = sc.pl.umap(methyl, color=["celltype", "batch"], return_fig=True, wspace=0.6)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/methyl_umap.png")
        plt.close()

    # add any pseudobulk loops that are in the scHi-C variable features but weren't loaded before
    # extra_loops = bulk.pixels(join=False)[:]
    # extra_loops['chr1'] = extra_loops['bin1_id'].map(chr_map)
    # extra_loops['chr2'] = extra_loops['bin2_id'].map(chr_map)
    # extra_loops['start1'] = extra_loops['bin1_id'].map(start_map).astype(int)
    # extra_loops['start2'] = extra_loops['bin2_id'].map(start_map).astype(int)
    # extra_loops['end1'] = extra_loops['bin1_id'].map(end_map).astype(int)
    # extra_loops['end2'] = extra_loops['bin2_id'].map(end_map).astype(int)
    # if use_ice:
    #     weight_map = frags['weight'].to_dict()
    #     extra_loops['weight1'] = extra_loops['bin1_id'].map(weight_map)
    #     extra_loops['weight2'] = extra_loops['bin2_id'].map(weight_map)
    #     extra_loops['oe'] = extra_loops['count'] * extra_loops['weight1'] * extra_loops['weight2']
    #     extra_loops['rank'] = extra_loops['oe'].rank(pct=True)
    # else:
    #     extra_loops['rank'] = extra_loops['count'].rank(pct=True)
    # extra_loops.dropna(inplace=True)
    # print(extra_loops)
    # loop_dfs = []
    # print('Filtering top loops in each chromosome...')
    # for chr_name in tqdm(sorted_nicely(bulk.chromnames)):
    #     chr_loops = extra_loops[(extra_loops['chr1'] == chr_name) & (extra_loops['chr2'] == chr_name)].copy()
    #     chr_schic = hic.var[hic.var['chrom'] == chr_name].copy()
    #     chr_schic['interaction'] = chr_schic.apply(lambda row: f"{row['chromStart']}-{row['chromEnd']},{row['target_chromStart']}-{row['target_chromEnd']}", axis=1)
    #     chr_loops['interaction'] = chr_loops.apply(lambda row: f"{row['start1']}-{row['end1']},{row['start2']}-{row['end2']}", axis=1)
    #     print(chr_schic)
    #     in_schic = chr_loops['interaction'].isin(chr_schic['interaction'])
    #     chr_loops = chr_loops[in_schic].copy()
    #     chr_loops['rank'] = chr_loops['rank'].rank(pct=True)
    #     # scale rank to (0.5, 1) since the graph decoder uses sigmoid
    #     chr_loops['rank'] = chr_loops['rank'] * 0.5 + 0.5
    #     chr_loops.reset_index(drop=True, inplace=True)
    #     chr_loops.drop(columns=['chr1', 'chr2', 'start1', 'start2', 'end1', 'end2', 'interaction'], inplace=True)
    #     if use_ice:
    #         chr_loops.drop(columns=['weight1', 'weight2', 'oe'], inplace=True)
    #     print(chr_loops)
    #     loop_dfs.append(chr_loops)
    # extra_loops = pd.concat(loop_dfs).reset_index(drop=True)
    # print(extra_loops)
    # loops = pd.concat([loops, extra_loops]).drop_duplicates(subset=['bin1_id', 'bin2_id']).reset_index(drop=True)
    # print(loops)

    
    genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
    diagonal_mask = hic.var_names.map(lambda s: s[-2] != '-' and s[-3] != '-')
    diagonal_anchors = hic.var[diagonal_mask]
    peaks = scglue.genomics.Bed(diagonal_anchors.assign(name=hic.var_names[diagonal_mask]))
    tss = genes.strand_specific_start_site()
    promoters = tss.expand(2000, 0)
    if atac_file is not None:
        atac_peaks = scglue.genomics.Bed(atac_peaks.assign(name=atac_peaks['name']))
    if methyl_file is not None: 
        methyl_genes = scglue.genomics.Bed(methyl_genes.assign(name=methyl_genes['name']))
        methyl_tss = methyl_genes.strand_specific_start_site()
        methyl_promoters = methyl_tss.expand(2000, 0)

    

    frags['peak_name'] = frags.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", axis=1)
    sign_dict = {}
    def sign_map(interval):
        if interval in sign_dict.keys():
            print(interval, sign_dict[interval])
            return sign_dict[interval]
        else:
            return 1

    overlap_graph = scglue.genomics.window_graph(
        promoters, peaks, 0,
        attr_fn=lambda l, r, d: {
            "weight": 1.0,
            "type": "overlap",
            "sign": sign_map(r.name)
        }
    )
    overlap_graph = nx.DiGraph(overlap_graph)
    print('Overlap #edges:', overlap_graph.number_of_edges())
    if atac_file is not None:
        atac_overlap_graph = scglue.genomics.window_graph(
            promoters, atac_peaks, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": sign_map(r.name)
            }
        )
        atac_hic_overlap_graph = scglue.genomics.window_graph(
            atac_peaks, peaks, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": sign_map(r.name)
            }
        )
        print('ATAC graph:', atac_overlap_graph)
        print('ATAC-HiC graph:', atac_hic_overlap_graph)
    if methyl_file is not None:
        methyl_overlap_graph = scglue.genomics.window_graph(
            promoters, methyl_promoters, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": -1
            }
        )
        methyl_hic_overlap_graph = scglue.genomics.window_graph(
            methyl_promoters, peaks, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": -1
            }
        )
        print('Methyl graph:', methyl_overlap_graph)
        print('Methyl-HiC graph:', methyl_hic_overlap_graph)

    dist_graph = scglue.genomics.window_graph(
        promoters, peaks, 150000,
        attr_fn=lambda l, r, d: {
            "dist": abs(d) / 2e6,
            "weight": scglue.genomics.dist_power_decay(abs(d)),
            "type": "dist",
            "sign": sign_map(r.name)
        }
    )
    dist_graph = nx.DiGraph(dist_graph)
    print('Distance #edges:', dist_graph.number_of_edges())

    bait_oe = loops[['bin1_id', 'bin2_id', 'rank']]
    if loops_offset != 0:
        bait_oe['bin2_id'] -= loops_offset
    

    seq_loops = pd.DataFrame()
    sorted_frags = frags.sort_values(by=['chrom', 'chromStart'])
    
    bait_oe['bin1_id'] = bait_oe['bin1_id'].astype(str)
    bait_oe['bin2_id'] = bait_oe['bin2_id'].astype(str)
    bait_oe = bait_oe.dropna().reset_index(drop=True)

    peak_map = frags.set_index('name')['peak_name'].to_dict()
    start_map = frags.set_index('name')['chromStart'].to_dict()
    end_map = frags.set_index('name')['chromEnd'].to_dict()

    #peaks = frags.set_index('peak_name')
    bait_oe['peak1'] = bait_oe['bin1_id'].map(peak_map)
    bait_oe['peak2'] = bait_oe['bin2_id'].map(peak_map)
    print(bait_oe)
    bait_oe = bait_oe.dropna().reset_index(drop=True)
    bait_oe['start1'] = bait_oe['bin1_id'].map(start_map).astype(int)
    bait_oe['start2'] = bait_oe['bin2_id'].map(start_map).astype(int)
    bait_oe['end1'] = bait_oe['bin1_id'].map(end_map).astype(int)
    bait_oe['end2'] = bait_oe['bin2_id'].map(end_map).astype(int)
    bait_oe['mid1'] = (bait_oe['end1'] - bait_oe['start1']).abs()
    bait_oe['mid2'] = (bait_oe['end2'] - bait_oe['start2']).abs()
    bait_oe['dist'] = (bait_oe['mid1'] - bait_oe['mid2']).abs() / 2e6
    bait_oe['peak1'] = bait_oe['peak1'].astype(str)
    bait_oe['peak2'] = bait_oe['peak2'].astype(str)

    bait_oe = bait_oe[['peak1', 'peak2', 'rank', 'dist']]

    frags.index = frags['peak_name']
    frags = scglue.genomics.Bed(frags)
    bait_oe.rename(columns={'rank': 'weight'}, inplace=True)
    pchic_graph = nx.from_pandas_edgelist(bait_oe, source="peak1", target="peak2", edge_attr=True, create_using=nx.DiGraph)
    

    nx.set_edge_attributes(pchic_graph, "hic", "type")
    nx.set_edge_attributes(pchic_graph, 1, "sign")
    print('Hi-C #edges:', pchic_graph.number_of_edges())

    gene_bait = scglue.genomics.window_graph(promoters, frags, 1000)

    chrom_attr = {}
    pos_attr = {}
    type_attr = {}

    rna.var["in_hic"] = biadjacency_matrix(gene_bait, genes.index).sum(axis=1) != 0
    print('Genes in Hi-C', rna.var["in_hic"].sum())

    o_prior = overlap_graph.copy()

    hvg_reachable = scglue.graph.reachable_vertices(o_prior, rna.var.query("highly_variable").index)

    hic.var["o_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print('Overlap variable Hi-C features', hic.var["o_highly_variable"].sum())

    o_prior = scglue.graph.compose_multigraph(o_prior, o_prior.reverse())
    for item in itertools.chain(hic.var_names, rna.var_names):
        try:
            if item[-2] == '-' and item.startswith('chr'):
                continue
        except Exception as e:
            pass
        o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
    nx.set_edge_attributes(o_prior, 1, "sign")
    nx.set_edge_attributes(o_prior, 0.0, "dist")
    o_prior = o_prior.subgraph(hvg_reachable)
    if atac_file is not None:
        atac_o_prior = atac_overlap_graph.copy()
        atac_hic_o_prior = atac_hic_overlap_graph.copy()
        # first limit to the highly variable ATAC peaks
        atac_reachable = scglue.graph.reachable_vertices(atac_hic_o_prior, atac.var.query("highly_variable").index)
        atac_hic_o_prior = atac_hic_o_prior.subgraph(atac_reachable)

        atac_o_prior = scglue.graph.compose_multigraph(atac_o_prior, atac_o_prior.reverse())
        atac_hic_o_prior = scglue.graph.compose_multigraph(atac_hic_o_prior, atac_hic_o_prior.reverse())
        atac_hic_o_prior = scglue.graph.compose_multigraph(atac_o_prior, atac_hic_o_prior)
        #atac_hic_o_prior = scglue.graph.compose_multigraph(atac_hic_o_prior, pchic_graph)
        # for item in itertools.chain(atac.var_names):
        #     atac_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #     atac_hic_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #atac_reachable = scglue.graph.reachable_vertices(atac_o_prior, rna.var.query("highly_variable").index)
        atac_reachable_hvgs = scglue.graph.reachable_vertices(atac_hic_o_prior, rna.var.query("highly_variable").index)
        
        #atac.var["highly_variable"] = [item in atac_reachable for item in atac.var_names]
        #atac_o_prior = atac_o_prior.subgraph(atac_reachable)
        atac_o_prior = atac_hic_o_prior.subgraph(atac_reachable_hvgs)
        # remove duplicate edges
        atac_o_prior = scglue.graph.compose_multigraph(atac_o_prior, atac_o_prior.reverse())
        for item in itertools.chain(atac.var_names):
            atac_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        # remove nodes that are note connected to either a gene or a Hi-C node
        remove_nodes = []
        for node in tqdm(atac_o_prior.nodes):
            if node in atac.var_names:
                remove = True
                for neighbor in atac_o_prior.neighbors(node):
                    if neighbor in rna.var_names or neighbor in hic.var_names:
                        remove = False
                        break
                if remove:
                    remove_nodes.append(node)
        atac_o_prior.remove_nodes_from(remove_nodes)
        print('ATAC overlap graph:', atac_o_prior)
    if methyl_file is not None:
        methyl_o_prior = methyl_overlap_graph.copy()
        methyl_hic_o_prior = methyl_hic_overlap_graph.copy()
        # first limit to the highly variable methyl peaks
        methyl_reachable = scglue.graph.reachable_vertices(methyl_hic_o_prior, methyl.var.query("highly_variable").index)
        methyl_hic_o_prior = methyl_hic_o_prior.subgraph(methyl_reachable)

        methyl_o_prior = scglue.graph.compose_multigraph(methyl_o_prior, methyl_o_prior.reverse())
        methyl_hic_o_prior = scglue.graph.compose_multigraph(methyl_hic_o_prior, methyl_hic_o_prior.reverse())
        methyl_hic_o_prior = scglue.graph.compose_multigraph(methyl_o_prior, methyl_hic_o_prior)
        #methyl_hic_o_prior = scglue.graph.compose_multigraph(methyl_hic_o_prior, pchic_graph)
        # for item in itertools.chain(methyl.var_names):
        #     methyl_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #     methyl_hic_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #methyl_reachable = scglue.graph.reachable_vertices(methyl_o_prior, rna.var.query("highly_variable").index)
        methyl_reachable_hvgs = scglue.graph.reachable_vertices(methyl_hic_o_prior, rna.var.query("highly_variable").index)
        
        #methyl.var["highly_variable"] = [item in methyl_reachable for item in methyl.var_names]
        #methyl_o_prior = methyl_o_prior.subgraph(methyl_reachable)
        methyl_o_prior = methyl_hic_o_prior.subgraph(methyl_reachable_hvgs)
        # remove duplicate edges
        methyl_o_prior = scglue.graph.compose_multigraph(methyl_o_prior, methyl_o_prior.reverse())
        for item in itertools.chain(methyl.var_names):
            methyl_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        # remove nodes that are note connected to either a gene or a Hi-C node
        remove_nodes = []
        for node in tqdm(methyl_o_prior.nodes):
            if node in methyl.var_names:
                remove = True
                for neighbor in methyl_o_prior.neighbors(node):
                    if neighbor in rna.var_names or neighbor in hic.var_names:
                        remove = False
                        break
                if remove:
                    remove_nodes.append(node)
        methyl_o_prior.remove_nodes_from(remove_nodes)
        print('Methyl overlap graph:', methyl_o_prior)

    d_prior = dist_graph.copy()

    hvg_reachable = scglue.graph.reachable_vertices(d_prior, rna.var.query("highly_variable").index)

    hic.var["d_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print('Distance variable Hi-C features', hic.var["d_highly_variable"].sum())

    d_prior = scglue.graph.compose_multigraph(d_prior, d_prior.reverse())
    for item in itertools.chain(hic.var_names, rna.var_names):
        try:
            if item[-2] == '-' and item.startswith('chr'):
                continue
        except Exception as e:
            pass
        d_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)

    nx.set_edge_attributes(d_prior, 1, "sign")
    d_prior = d_prior.subgraph(hvg_reachable)
    print(pchic_graph)
    pchic_graph = scglue.graph.compose_multigraph(pchic_graph, pchic_graph.reverse())
    
    print('Composing overlap graph with Hi-C graph...')
    dcq_prior = scglue.graph.compose_multigraph(o_prior, pchic_graph)
    nx.set_edge_attributes(dcq_prior, 1, "sign")
    if atac_file is not None:
        print('Composing ATAC overlap graph with Hi-C graph...')
        dcq_prior = scglue.graph.compose_multigraph(dcq_prior, atac_o_prior)
    if methyl_file is not None:
        print('Composing Methyl overlap graph with Hi-C graph...')
        dcq_prior = scglue.graph.compose_multigraph(dcq_prior, methyl_o_prior)

    hvg_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)
    hic.var["dcq_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    if atac_file is not None:
        atac_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)
        # atac.var["highly_variable"] = [item in atac_reachable for item in atac.var_names]
        atac.var["highly_variable"] = [item in atac_reachable for item in atac.var_names]
        print('ATAC variable Hi-C features', atac.var["highly_variable"].sum())
    if methyl_file is not None:
        methyl_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)
        # methyl.var["highly_variable"] = [item in methyl_reachable for item in methyl.var_names]
        methyl.var["highly_variable"] = [item in methyl_reachable for item in methyl.var_names]
        print('Methyl variable Hi-C features', methyl.var["highly_variable"].sum())
    keep_distal_mask = hic.var_names.map(lambda s: s[-2] == '-' or s[-3] == '-')
    # set distal entries as highly variable too
    hic.var["dcq_highly_variable"] = hic.var["dcq_highly_variable"] | keep_distal_mask  
    # and keep all roots of distal entries if they aren't already 
    distal_source = hic.var_names[keep_distal_mask].map(lambda s: s.rsplit('-', 1)[0]).values
    keep_non_distal_mask = hic.var_names.isin(distal_source)
    hic.var["dcq_highly_variable"] = hic.var["dcq_highly_variable"] | keep_non_distal_mask
    print('Full Hi-C prior variable features:', hic.var["dcq_highly_variable"].sum())

    for item in dcq_prior.nodes:
        if dcq_prior.has_edge(item, item):
            continue
        else:
            dcq_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)

    chrom_attr = {}
    strata_attr = {}
    pos_attr = {}
    type_attr = {}
    for n in tqdm(dcq_prior.nodes):
        if n in genes.index:
            type_attr[n] = 'RNA'
            row = genes.loc[n]
            strata_attr[n] = 0
        elif n in frags.index:
            type_attr[n] = 'Hi-C'
            row = frags.loc[n]
            strata_attr[n] = 0.0
        else:
            if atac_file is not None and n in atac.var_names:
                type_attr[n] = 'ATAC'
                row = atac_peaks.loc[n]
                strata_attr[n] = 0.0
            elif methyl_file is not None and n in methyl.var_names:
                type_attr[n] = 'Methyl'
                row = methyl_genes.loc[n]
                strata_attr[n] = 0.0
            else:
                type_attr[n] = 'Hi-C'
                chr_split = str(n).split(':')
                row = {}
                row['chrom'] = str(chr_split[0])
                try:
                    pos = chr_split[1].split('-')
                    row['chromStart'] = int(pos[0])
                except Exception as e:
                    print(n, chr_split, pos, e)
                    row['chromStart'] = 0
                try:
                    strata_attr[n] = int(str(n).split('-')[-1])
                except Exception as e:
                    print(e)
                    strata_attr[n] = 0
        chrom_attr[n] = str(row['chrom'])
        pos_attr[n] = int(row['chromStart'])

    edge_ids = {}
    for edge_i, e in enumerate(dcq_prior.edges):
        edge_ids[e] = edge_i
    nx.set_edge_attributes(dcq_prior, edge_ids, "edge_id")

    if use_ice:
        suffix = f'ice_{loop_q}'
    elif use_raw_pseudobulk:
        suffix = f'raw_{loop_q}'
    else:
        suffix = f'deeploop_{loop_q}'
    suffix += f'_2d_{n_strata}'
    print(suffix)
    os.makedirs(os.path.join(out_dir, 'rna'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'hic'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'atac'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'methyl'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'graphs'), exist_ok=True)

    hic.write(f"{out_dir}/hic/hic_{resolution}_{suffix}.h5ad", compression="gzip")
    rna.write(f"{out_dir}/rna/rna_{resolution}_{suffix}.h5ad", compression="gzip")
    if atac_file is not None:
        atac.write(f"{out_dir}/atac/atac_{resolution}_{suffix}.h5ad", compression="gzip")
    if methyl_file is not None:
        methyl.write(f"{out_dir}/methyl/methyl_{resolution}_{suffix}.h5ad", compression="gzip")

    nx.set_node_attributes(dcq_prior, chrom_attr, "chrom")
    nx.set_node_attributes(dcq_prior, pos_attr, "chrom_pos")
    nx.set_node_attributes(dcq_prior, type_attr, "feature_type")
    nx.write_graphml(dcq_prior, f"{out_dir}/graphs/dcq_prior_{resolution}_{suffix}.graphml.gz", edge_id_from_attribute='edge_id', named_key_ids=True)