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
                
                mat = matrix.A
            
            else:
                mat = np.zeros((len(chr_anchors), len(chr_anchors)))
            chr_mats.append(mat)
        mat = csr_matrix(block_diag(*chr_mats))
    else:
        mat = c.matrix(sparse=True).fetch(chr_only)
    return cell_i, cell, mat



def get_flattened_matrices(dataset, n_strata, preprocessing=None, agg_fn=None, chr_only=None, offset=0):
    mats = {}
    results = []
    with Pool(7) as p:
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
        strata_mask = np.zeros_like(full_mats[0].A)
        for k in range(n_strata):
            strata_mask += np.eye(strata_mask.shape[0], k=k, dtype=full_mats[0].dtype)
        mat = []
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            tmp_mat = full_mats[cell_i].A
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
    loop_q = args.loop_q
    load_rna = args.load_rna
    load_hic = False
    use_toploops = False 
    use_ice = args.use_ice
    use_2d_rep = args.use_2d
    viz_rna = args.viz_rna
    use_raw_pseudobulk = True
    use_compartment_signs = False
    if use_compartment_signs:
        import bioframe
    loops_offset = args.offset
    min_count = args.min_count
    n_distal_interactions = args.distal_interactions
    filter_strata = args.filter_strata
    n_genes = args.n_genes
    gene_list = args.gene_list
    bulk_rna_sampling = args.bulk_rna_sampling
    n_samples_each = args.bulk_n_samples
    counts_per_cell = args.bulk_n_counts
    bulk_hic = args.bulk_hic
    n_loops = 1000000 
    if not use_toploops:
        loop_q = float(loop_q)
    else:
        if loop_q.endswith('k'):
            n_loops = int(loop_q[:-1]) * 1000
    dataset_name = args.dset
    out_dir = f'{dataset_name}_data'
    plot_dir = f'{dataset_name}_plots'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'rna'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'graphs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'hic'), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print('Parsing SCORE args...')
    parser = argparse.ArgumentParser()
    sys.argv = sys.argv[glue_args + 1:]
    print(sys.argv)
    args, x, y, depths, batches, dataset, valid_dataset = parse_args(parser)

    # dataset.write_binned_scool(f'{dataset_name}_100kb.scool', factor=2, new_res_name='100kb')
    # sys.exit(0)

    if bulk_hic is not None:
        bulk = cooler.Cooler(bulk_hic)
    else:
        cool_files = cooler.fileops.list_scool_cells(dataset.scool_file)
        cool_files = [f"{dataset.scool_file}::{cell}" for cell in cool_files]
        os.makedirs('scools', exist_ok=True)
        out_cool_file = f"scools/{dataset_name}_{resolution}_bulk.cool"

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

    if use_compartment_signs:
        bins = bulk.bins()[:]
        if 'weight' not in bins.columns:
            set_verbosity_level(1)
            cooler.balance_cooler(bulk, cis_only=True, store=True)
            bins = bulk.bins()[:]
        hg19_genome = bioframe.load_fasta('/mnt/jinstore/JinLab01/LAB/Genome_references/cellranger_atac_hg19ref/hg19/fasta/genome.fa')
        gc_cov = bioframe.frac_gc(bins[['chrom', 'start', 'end']], hg19_genome)
        gc_cov.to_csv(f'hg19_gc_cov_{resolution}.tsv', index=False, sep='\t')
        print(gc_cov)
        # make dummy uniform bins for cooler and gc_cov
        dummy_bins = []
        for chrom in bulk.chromnames:
            chrom_bins = bins[bins['chrom'] == chrom].copy()
            chrom_bins['start'] = np.arange(0, len(chrom_bins) * int(resolution[:-2]), int(resolution[:-2]))
            chrom_bins['end'] = chrom_bins['start'] + int(resolution[:-2])
            dummy_bins.append(chrom_bins)
        dummy_bins = pd.concat(dummy_bins)
        gc_cov['start'] = dummy_bins['start']
        gc_cov['end'] = dummy_bins['end']
        
        dummy_pixels = bulk.pixels()[:]
        dummy_pixels['bin2_id'] = dummy_pixels['bin2_id'] - loops_offset
        cooler.create_cooler(f'{dataset_name}_dummy.cool', dummy_bins[['chrom', 'start', 'end', 'weight']], dummy_pixels)
        dummy_cool = cooler.Cooler(f'{dataset_name}_dummy.cool')
        set_verbosity_level(1)
        cooler.balance_cooler(bulk, cis_only=True, store=True)
        view_df = pd.DataFrame({'chrom': dummy_cool.chromnames,
                        'start': 0,
                        'end': dummy_cool.chromsizes.values,
                        'name': dummy_cool.chromnames}
                    )
        # obtain first 3 eigenvectors
        cis_eigs = cooltools.eigs_cis(
                                dummy_cool,
                                gc_cov,
                                n_eigs=3,
                                view_df=view_df
                                )
        eigenvector_track = cis_eigs[1][['chrom','start','end','E1', 'E2']]

        f, ax = plt.subplots(
            figsize=(15, 10),
        )

        norm = LogNorm(vmax=0.1)
        viz_chr = 'chr10'
        viz_chr_size = len(dummy_cool.bins().fetch(viz_chr))
        im = ax.matshow(
            dummy_cool.matrix().fetch(viz_chr),
            norm=norm,
            cmap='OrRd'
        )
        plt.axis([0,viz_chr_size,viz_chr_size,0])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='corrected frequencies')
        ax.set_ylabel(viz_chr)
        ax.xaxis.set_visible(False)

        ax1 = divider.append_axes("top", size="20%", pad=0.25, sharex=ax)
        ax1.plot([0,viz_chr_size],[0,0],'k',lw=0.25)
        ax1.plot( eigenvector_track['E1'].values[:viz_chr_size], label='E1')

        ax1.set_ylabel('E1')
        ax1.set_xticks([])

        # for i in np.where(np.diff( (cis_eigs[1]['E1']>0).astype(int)))[0]:
        #     ax.plot([0, viz_chr_size],[i,i],'k',lw=0.5)
        #     ax.plot([i,i],[0, viz_chr_size],'k',lw=0.5)
        plt.savefig(f'{plot_dir}/eigenvector.png')
        plt.close()

        compartment_signs = np.sign(cis_eigs[1]['E1'].values)
        compartment_signs = np.nan_to_num(compartment_signs, nan=-1)  # empty regions are assumed inactive
        print(compartment_signs)

    frags = bulk.bins()[:]
    if use_compartment_signs:
        frags['compartment'] = np.array([int(c) for c in compartment_signs])
    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)
    frags['name'] = frags.index.astype(str)
    loops = bulk.pixels(join=False)[:]
    if use_ice:
        weight_map = frags['weight'].to_dict()
        loops['weight1'] = loops['bin1_id'].map(weight_map)
        loops['weight2'] = loops['bin2_id'].map(weight_map)
        loops['oe'] = loops['count'] * loops['weight1'] * loops['weight2']
        loops['rank'] = loops['oe'].rank(pct=True)
    else:
        loops['rank'] = loops['count'].rank(pct=True)
    loops.dropna(inplace=True)
    chr_map = frags['chrom'].to_dict()
    loops['chr1'] = loops['bin1_id'].map(chr_map)   
    loops['chr2'] = loops['bin2_id'].map(chr_map)
    loop_dfs = []
    print('Filtering top loops in each chromosome...')
    for chr_name in tqdm(sorted_nicely(bulk.chromnames)):
        chr_loops = loops[(loops['chr1'] == chr_name) & (loops['chr2'] == chr_name)].copy()
        chr_loops['rank'] = chr_loops['oe' if use_ice else 'count'].rank(pct=True)
        loop_cutoff = np.quantile(chr_loops['rank'].values, q=loop_q)
        chr_loops = chr_loops.loc[chr_loops['rank'] >= loop_cutoff].copy()
        chr_loops['rank'] = chr_loops['rank'].rank(pct=True)
        # scale rank to (0.5, 1) since the graph decoder uses sigmoid
        chr_loops['rank'] = chr_loops['rank'] * 0.5 + 0.5
        chr_loops.reset_index(drop=True, inplace=True)
        chr_loops.drop(columns=['chr1', 'chr2'], inplace=True)
        loop_dfs.append(chr_loops)
    loops = pd.concat(loop_dfs).reset_index(drop=True)
    # if bulk_hic is not None:  # add diagonal signal connecting each bin to its neighbor with a weight of 1
    #     print('Adding diagonal signal...')
    #     a1 = bulk.bins()[:].index[:-1]
    #     a2 = bulk.bins()[:].index[1:]
    #     diag = pd.DataFrame({'bin1_id': a1, 'bin2_id': a2, 'count': 1, 'rank': 1.0})
    #     loops = pd.concat([loops, diag], ignore_index=True)
    print(loops)

    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)

    if use_2d_rep:
        base_rna_filename = 'rna_base_2d.h5ad'
    else:
        base_rna_filename = 'rna_base.h5ad'
    if  base_rna_filename not in os.listdir(os.path.join(out_dir, 'rna')) or not load_rna:
        rna = ad.read_h5ad(rna_file)
        try:
            rna.X = rna.layers["counts"]
        except Exception as e:
            pass
        
        if 'batch' not in rna.obs.columns:
            rna.obs['batch'] = 0
        rna.layers["counts"] = rna.X.copy()
        # reset rna vars
        if 'human_brain' in dataset_name:
            try:
                rna.obs['celltype'] = rna.obs['supercluster_term']
            except Exception as e:
                pass
            try:
                rna.obs['batch'] = rna.obs['donor_id']
            except Exception as e:
                pass
            rna.layers["counts"] = rna.X.copy()
            try:
                dup_genes = rna.var['Gene'].duplicated(keep='first')
            except Exception as e:
                try:
                    dup_genes = rna.var['feature_name'].duplicated(keep='first')
                except Exception as e:
                    pass
            try:
                rna = rna[:, ~dup_genes]
            except Exception as e:
                pass
            try:
                rna.var.set_index('Gene', inplace=True)
            except Exception as e:
                try:
                    rna.var.set_index('feature_name', inplace=True)
                except Exception as e:
                    pass
            try:
                rna.var_names = rna.var.index
                rna.var = pd.DataFrame(index=rna.var_names)
            except Exception as e:
                pass
        if bulk_rna_sampling:  # train model on sampled bulk RNA instead of real scRNA-seq
            # merge each celltype into psuedobulk samples then sample from these
            rna.obs['celltype'] = rna.obs['celltype'].astype(str)
            celltype_profiles = {}
            celltype_n_cells = {}
            for celltype in pd.unique(rna.obs['celltype']):
                celltype_cells = rna[rna.obs['celltype'] == celltype].copy()
                celltype_counts = np.array(celltype_cells.layers['counts'].sum(axis=0)).squeeze()
                mean_count_per_cell = np.mean(celltype_cells.layers['counts'].sum(axis=1))
                print(celltype, mean_count_per_cell)
                celltype_profiles[celltype] = celltype_counts.squeeze()
                celltype_n_cells[celltype] = int(celltype_cells.X.shape[0] / 2)
            new_rna_x = []
            new_celltypes = []
            for celltype in pd.unique(rna.obs['celltype']):
                for i in range(celltype_n_cells[celltype]):
                    celltype_counts = celltype_profiles[celltype]
                    cell_sample_idxs = np.random.choice(np.arange(len(celltype_counts)), size=counts_per_cell, p=celltype_counts / np.sum(celltype_counts))
                    cell_sample = np.zeros(len(celltype_profiles[celltype]))
                    cell_sample[cell_sample_idxs] += 1
                    new_rna_x.append(cell_sample)
                    new_celltypes.append(celltype)
            new_rna_x = np.uint8(new_rna_x)
            rna = ad.AnnData(new_rna_x, var=rna.var, dtype=np.uint8)
            rna.obs['celltype'] = new_celltypes
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
        if not rna.var['chrom'].iloc[0].startswith('chr'):
            rna.var['chrom'] = 'chr' + rna.var['chrom']
        drop_cols = []
        for col in rna.var.columns:
            if col not in keep_columns:
                drop_cols.append(col)
        rna.var.drop(columns=drop_cols, inplace=True)
        rna = rna[:, rna.var['chrom'].notna()].copy()
        genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
        rna.write(f"{out_dir}/rna/{base_rna_filename}", compression="gzip")
    else:
        rna = ad.read_h5ad(f"{out_dir}/rna/{base_rna_filename}")
    
    sc.pp.filter_genes(rna, min_counts=1)
    print('Embedding RNA...')
    print('Highly variable genes...')
    sc.pp.highly_variable_genes(rna, n_top_genes=n_genes, flavor="seurat_v3")
    if gene_list is not None:
        for gene in gene_list:
            if gene in rna.var_names:
                print(gene)
                rna.var.loc[gene, 'highly_variable'] = True
    # # #sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print("Normalize")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    print("PCA")
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    if viz_rna:
        sc.pp.neighbors(rna, n_pcs=100, metric="cosine")
        sc.tl.umap(rna)

        fig = sc.pl.umap(rna, color=["celltype", "batch"], return_fig=True, wspace=0.6)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/rna_umap.png")
        plt.close()

    if use_2d_rep:
        base_hic_filename = f"hic_base_{resolution}_2d.h5ad"
    else:
        base_hic_filename = f"hic_base_{resolution}.h5ad"
    base_hic_path = f"{out_dir}/hic/{base_hic_filename}"
    if base_hic_filename not in os.listdir(os.path.join(out_dir, 'hic')) or not load_hic:
        print('Loading scHi-C sparse matrices...')
        mats = dataset.get_sparse_matrices()
        mat = []
        strata_adatas = []
        total_interactions = 0
        print(f'Processing strata...')
        #visibility_mat = dataset.write_cell_bin_matrix()
        for k in range(abs(loops_offset), n_strata + abs(loops_offset)):
            strata_mat = []
            for cell_i, cell in enumerate(sorted(dataset.cell_list)):
                # TODO: maybe diagonal should be cis visibility?
                # if k == 0:
                #     new_strata = visibility_mat[cell_i]
                #     strata_mat.append(new_strata)
                # else:
                new_strata = list(mats[cell_i].diagonal(k=k))
                if len(new_strata) < len(frags):
                    new_strata += [0] * (len(frags) - len(new_strata))
                if resolution == '10kb' or resolution == '100kb':
                    strata_mat.append(np.uint8(new_strata))
                else:
                    strata_mat.append(new_strata)
            # create per-strata anndata
            strata_hic = ad.AnnData(np.array(strata_mat), dtype=np.uint8 if resolution in ['10kb', '20kb'] else np.int32)
            strata_hic.obs_names = sorted(dataset.cell_list)
            strata_hic.obs_names = strata_hic.obs_names.map(lambda s: s.replace(f'.{dataset.res_name}', ''))
            genomic_pos = dataset.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
            if k - loops_offset == 0:
                strata_hic.var_names = genomic_pos
            else:
                strata_hic.var_names = genomic_pos + f'-{k - loops_offset}'
            strata_hic.var['root'] = strata_hic.var_names.map(lambda s: s.rsplit('-', 1)[0] if (s[-2] == '-' or s[-3] == '-') else s)
            split = strata_hic.var['root'].str.split(r"[:-]")
            strata_hic.var["chrom"] = split.map(lambda x: x[0])
            strata_hic.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
            strata_hic.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
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
    sc.tl.pca(hic, n_comps=100, svd_solver="auto")
    sc.pp.neighbors(hic, n_pcs=100, metric="cosine")
    # scglue.data.lsi(hic, n_components=100, n_iter=50, n_oversamples=20)
    # sc.pp.neighbors(hic, use_rep="X_lsi", metric="cosine")
    sc.tl.umap(hic)
    fig = sc.pl.umap(hic, color=["celltype", "batch"], return_fig=True)
    fig.savefig(f'{plot_dir}/pfc_hic_umap_{resolution}.png')
    plt.close()
    
    genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
    diagonal_mask = hic.var_names.map(lambda s: s[-2] != '-' and s[-3] != '-')
    diagonal_anchors = hic.var[diagonal_mask]
    peaks = scglue.genomics.Bed(diagonal_anchors.assign(name=hic.var_names[diagonal_mask]))
    tss = genes.strand_specific_start_site()
    promoters = tss.expand(2000, 0)

    frags['peak_name'] = frags.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", axis=1)
    if use_compartment_signs:
        sign_dict = frags.set_index('peak_name')['compartment'].to_dict()
    else:
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
    # if use_2d_rep:
    #     frags['peak_name'] = frags.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}-0", axis=1)
    # else:

    peak_map = frags.set_index('name')['peak_name'].to_dict()
    start_map = frags.set_index('name')['chromStart'].to_dict()
    end_map = frags.set_index('name')['chromEnd'].to_dict()
    if use_compartment_signs:
        compartment_map = frags.set_index('name')['compartment'].to_dict()

    #peaks = frags.set_index('peak_name')
    bait_oe['peak1'] = bait_oe['bin1_id'].map(peak_map)
    bait_oe['peak2'] = bait_oe['bin2_id'].map(peak_map)
    bait_oe['start1'] = bait_oe['bin1_id'].map(start_map).astype(int)
    bait_oe['start2'] = bait_oe['bin2_id'].map(start_map).astype(int)
    bait_oe['end1'] = bait_oe['bin1_id'].map(end_map).astype(int)
    bait_oe['end2'] = bait_oe['bin2_id'].map(end_map).astype(int)
    bait_oe['mid1'] = (bait_oe['end1'] - bait_oe['start1']).abs()
    bait_oe['mid2'] = (bait_oe['end2'] - bait_oe['start2']).abs()
    bait_oe['dist'] = (bait_oe['mid1'] - bait_oe['mid2']).abs() / 2e6
    bait_oe['peak1'] = bait_oe['peak1'].astype(str)
    bait_oe['peak2'] = bait_oe['peak2'].astype(str)
    # if use_compartment_signs:
    #     bait_oe['compartment1'] = bait_oe['bin1_id'].map(compartment_map)
    #     bait_oe['compartment2'] = bait_oe['bin2_id'].map(compartment_map)
    #     # AA, AB, and BA are active, BB is inactive
    #     bait_oe['sign'] = np.int32(np.sign(bait_oe['compartment1'] * bait_oe['compartment2']))

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
            if use_2d_rep and item[-2] == '-' and item.startswith('chr'):
                continue
        except Exception as e:
            pass
        o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
    if not use_compartment_signs:
        nx.set_edge_attributes(o_prior, 1, "sign")
    nx.set_edge_attributes(o_prior, 0.0, "dist")

    o_prior = o_prior.subgraph(hvg_reachable)

    d_prior = dist_graph.copy()

    hvg_reachable = scglue.graph.reachable_vertices(d_prior, rna.var.query("highly_variable").index)

    hic.var["d_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print('Distance variable Hi-C features', hic.var["d_highly_variable"].sum())

    d_prior = scglue.graph.compose_multigraph(d_prior, d_prior.reverse())
    for item in itertools.chain(hic.var_names, rna.var_names):
        try:
            if use_2d_rep and item[-2] == '-' and item.startswith('chr'):
                continue
        except Exception as e:
            pass
        d_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
    if not use_compartment_signs:
        nx.set_edge_attributes(d_prior, 1, "sign")

    d_prior = d_prior.subgraph(hvg_reachable)
    print(pchic_graph)
    pchic_graph = scglue.graph.compose_multigraph(pchic_graph, pchic_graph.reverse())
    print('Composing overlap graph with Hi-C graph...')
    dcq_prior = scglue.graph.compose_multigraph(o_prior, pchic_graph)

    hvg_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)

    hic.var["dcq_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
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
    if not use_compartment_signs:
        nx.set_edge_attributes(dcq_prior, 1, "sign")
        

    edge_count = dcq_prior.number_of_edges()

    # for i, e in enumerate(dcq_prior.edges(data=True)):
    #     if i < 5:
    #         print(e)
    #     else:
    #         break

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

    # print(hic.var.head())
    # print(rna.var.head())

    if use_ice:
        suffix = f'ice_{loop_q}'
    elif use_raw_pseudobulk:
        suffix = f'raw_{loop_q}'
    else:
        suffix = f'deeploop_{loop_q}'
    if use_2d_rep:
        suffix += f'_2d_{n_strata}'
    print(suffix)
    os.makedirs(os.path.join(out_dir, 'rna'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'hic'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'graphs'), exist_ok=True)

    hic.write(f"{out_dir}/hic/hic_{resolution}_{suffix}.h5ad", compression="gzip")
    rna.write(f"{out_dir}/rna/rna_{resolution}_{suffix}.h5ad", compression="gzip")

    nx.set_node_attributes(dcq_prior, chrom_attr, "chrom")
    nx.set_node_attributes(dcq_prior, pos_attr, "chrom_pos")
    nx.set_node_attributes(dcq_prior, type_attr, "feature_type")
    nx.write_graphml(dcq_prior, f"{out_dir}/graphs/dcq_prior_{resolution}_{suffix}.graphml.gz", edge_id_from_attribute='edge_id', named_key_ids=True)