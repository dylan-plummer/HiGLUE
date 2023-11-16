import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import cooler
from matplotlib import rcParams
import os
import sys
import itertools
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from networkx.algorithms.bipartite import biadjacency_matrix
from scloop.sc_args import parse_args


if __name__ == '__main__':
    glue_parser = argparse.ArgumentParser()
    glue_parser.add_argument('--loop_q', type=str, default='0.99')
    glue_parser.add_argument('--n_strata', type=int, default=32)
    glue_parser.add_argument('--use_ice', action='store_true')

    glue_parser.add_argument('--rna_file', type=str, default=None)
    glue_parser.add_argument('--gtf', type=str, default=None)
    glue_parser.add_argument('--dset', type=str, default=None)
    glue_parser.add_argument('--scool', type=str, default=None)
    glue_parser.add_argument('--reference', type=str, default=None)
    glue_parser.add_argument('--resolution', type=str, default='100kb')
    glue_parser.add_argument('--min_depth', type=int, default=20000)

    args = glue_parser.parse_args()

    

    n_strata = args.n_strata
    resolution = args.resolution
    gtf_file = args.gtf
    rna_file = args.rna_file
    loop_q = args.loop_q
    load_rna = False
    load_hic = True
    use_toploops = False 
    use_ice = args.use_ice
    use_2d_rep = False
    use_raw_pseudobulk = True
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

    print('Parsing scloop args')
    parser = argparse.ArgumentParser()
    sys.argv = sys.argv[3:]
    args, x, y, depths, batches, dataset, valid_dataset = parse_args(parser)

    cool_files = []
    for cell in sorted(dataset.cell_list):
        cool_files.append(f"{dataset.scool_file}::/cells/{cell}")
    os.makedirs('scools', exist_ok=True)
    out_cool_file = f"scools/{dataset_name}_{resolution}_bulk.cool"

    try:
        bulk = cooler.Cooler(out_cool_file)
    except Exception as e:
        print('Could not find bulk cooler file, generating from psuedobulk...')
        cooler.merge_coolers(out_cool_file, cool_files, mergebuf=40000000)
        bulk = cooler.Cooler(out_cool_file)
        if use_ice:
            cooler.balance_cooler(bulk, cis_only=True, store=True)
            bulk = cooler.Cooler(out_cool_file)

    loops = bulk.pixels(join=False)[:]
    frags = bulk.bins()[:]
    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)
    frags['name'] = frags.index.astype(str)
    if use_ice:
        weight_map = frags['weight'].to_dict()
        loops['weight1'] = loops['bin1_id'].map(weight_map)
        loops['weight2'] = loops['bin2_id'].map(weight_map)
        loops['oe'] = loops['count'] * loops['weight1'] * loops['weight2']
        loops['rank'] = loops['oe'].rank(pct=True)
    else:
        loops['rank'] = loops['count'].rank(pct=True)
    loops.dropna(inplace=True)
    loop_cutoff = np.quantile(loops['rank'].values, q=loop_q)
    loops = loops.loc[loops['rank'] >= loop_cutoff].copy()
    loops['rank'] = loops['rank'].rank(pct=True)
    loops.reset_index(drop=True, inplace=True)
    print(loops)

    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)

    if use_2d_rep:
        base_rna_filename = 'rna_base_2d.h5ad'
    else:
        base_rna_filename = 'rna_base.h5ad'
    if  base_rna_filename not in os.listdir(os.path.join(out_dir, 'rna')) or not load_rna:
        rna = ad.read_h5ad(rna_file)
        print(rna)
        try:
            rna.X = rna.layers["counts"]
        except Exception as e:
            pass
        
        if 'batch' not in rna.obs.columns:
            rna.obs['batch'] = 0
        rna.layers["counts"] = rna.X.copy()
        try:
            rna.var.drop(columns=['chrom'], inplace=True)
            rna.var.drop(columns=['chromStart'], inplace=True)
            rna.var.drop(columns=['chromEnd'], inplace=True)
            rna.var.drop(columns=['strand'], inplace=True)
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
        rna.var['chrom'] = rna.var['chrom']
        rna = rna[:, rna.var['chrom'].notna()].copy()
        rna.write(f"{out_dir}/rna/{base_rna_filename}", compression="gzip")
    else:
        rna = ad.read_h5ad(f"{out_dir}/rna/{base_rna_filename}")

    sc.pp.filter_genes(rna, min_counts=1)

    print('Embedding RNA...')
    sc.pp.highly_variable_genes(rna, n_top_genes=10000, flavor="seurat_v3")
    # # #sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
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
        if use_2d_rep:
            mats = dataset.get_sparse_matrices()
            mat = []
            strata_k = []
            for cell_i, cell in tqdm(enumerate(sorted(dataset.cell_list)), total=len(dataset.cell_list)):
                counts = []
                strata = []
                for k in range(n_strata):
                    new_strata = list(mats[cell_i].diagonal(k=k))
                    counts += new_strata
                    if cell_i == 0:
                        strata_k += [k] * len(new_strata)
                mat.append(counts)
            mat = np.array(mat)
            print(mat.shape)
        else:
            mat = dataset.write_cell_bin_matrix(max_dist=n_strata)
        hic = ad.AnnData(mat)
        hic.obs_names = sorted(dataset.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{dataset.res_name}', ''))
        genomic_pos = dataset.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
        if use_2d_rep:
            new_var_names = pd.concat([genomic_pos.iloc[k:] + f'-{k}' for k in range(n_strata)])
            print(new_var_names)
        else:
            new_var_names = genomic_pos
        hic.var_names = new_var_names
        split = hic.var_names.str.split(r"[:-]")
        hic.var["chrom"] = split.map(lambda x: x[0])
        hic.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
        hic.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
        hic.var = hic.var.sort_values(by=['chrom', 'chromStart', 'chromEnd'])

        hic.layers["counts"] = hic.X.copy()

        hic.obs['celltype'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'cluster'] for cell in hic.obs_names])
        hic.obs['depth'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'depth'] for cell in hic.obs_names])
        hic.obs['batch'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'batch'] for cell in hic.obs_names])

        hic.obs['dataset'] = 'train'
        hic.write(base_hic_path, compression="gzip")
    else:
        hic = ad.read_h5ad(base_hic_path)
    print('Embedding Hi-C data...')
    sc.pp.filter_genes(hic, min_counts=3)
    sc.pp.normalize_per_cell(hic, counts_per_cell_after=1e5)
    sc.tl.pca(hic, n_comps=100, svd_solver="auto")
    sc.pp.neighbors(hic, n_pcs=100, metric="cosine")
    sc.tl.umap(hic)
    fig = sc.pl.umap(hic, color=["celltype", "batch"], return_fig=True)
    fig.savefig(f'{plot_dir}/pfc_hic_umap_{resolution}.png')
    plt.close()

    genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
    peaks = scglue.genomics.Bed(hic.var.assign(name=hic.var_names))
    tss = genes.strand_specific_start_site()
    promoters = tss.expand(2000, 0)

    overlap_graph = scglue.genomics.window_graph(
        genes.expand(2000, 0), peaks, 0,
        attr_fn=lambda l, r, d: {
            "weight": 1.0,
            "type": "overlap"
        }
    )
    overlap_graph = nx.DiGraph(overlap_graph)
    print('Overlap #edges:', overlap_graph.number_of_edges())

    dist_graph = scglue.genomics.window_graph(
        promoters, peaks, 150000,
        attr_fn=lambda l, r, d: {
            "dist": abs(d),
            "weight": scglue.genomics.dist_power_decay(abs(d)),
            "type": "dist"
        }
    )
    dist_graph = nx.DiGraph(dist_graph)
    print('Distance #edges:', dist_graph.number_of_edges())

    bait_oe = loops[['bin1_id', 'bin2_id', 'rank']]

    seq_loops = pd.DataFrame()
    sorted_frags = frags.sort_values(by=['chrom', 'chromStart'])
    
    bait_oe['bin1_id'] = bait_oe['bin1_id'].astype(str)
    bait_oe['bin2_id'] = bait_oe['bin2_id'].astype(str)
    bait_oe = bait_oe.dropna().reset_index(drop=True)
    if use_2d_rep:
        frags['peak_name'] = frags.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}-0", axis=1)
    else:
        frags['peak_name'] = frags.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", axis=1)
 
    peak_map = frags.set_index('name')['peak_name'].to_dict()
    start_map = frags.set_index('name')['chromStart'].to_dict()
    end_map = frags.set_index('name')['chromEnd'].to_dict()

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
 
    print(bait_oe)
    bait_oe = bait_oe[['peak1', 'peak2', 'rank', 'dist']]


    frags.index = frags['peak_name']
    frags = scglue.genomics.Bed(frags)
    print(frags)
    bait_oe.rename(columns={'rank': 'weight'}, inplace=True)
    pchic_graph = nx.from_pandas_edgelist(bait_oe, source="peak1", target="peak2", edge_attr=True, create_using=nx.DiGraph)

    nx.set_edge_attributes(pchic_graph, "hic", "type")
    nx.set_edge_attributes(pchic_graph, 1, "sign")
    print('Hi-C #edges:', pchic_graph.number_of_edges())

    gene_bait = scglue.genomics.window_graph(promoters, frags, 1000)

    chrom_attr = {}
    pos_attr = {}
    type_attr = {}


    rna.var["in_hic"] = biadjacency_matrix(gene_bait, genes.index).sum(axis=1).A1 != 0
    print('Genes in Hi-C', rna.var["in_hic"].sum())

    o_prior = overlap_graph.copy()

    hvg_reachable = scglue.graph.reachable_vertices(o_prior, rna.var.query("highly_variable").index)

    hic.var["o_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print('Overlap variable Hi-C features', hic.var["o_highly_variable"].sum())

    o_prior = scglue.graph.compose_multigraph(o_prior, o_prior.reverse())
    for item in itertools.chain(hic.var_names, rna.var_names):
        o_prior.add_edge(item, item, weight=1.0, type="self-loop")
    nx.set_edge_attributes(o_prior, 1, "sign")
    nx.set_edge_attributes(o_prior, 0.0, "dist")

    o_prior = o_prior.subgraph(hvg_reachable)

    d_prior = dist_graph.copy()

    hvg_reachable = scglue.graph.reachable_vertices(d_prior, rna.var.query("highly_variable").index)

    hic.var["d_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print('Distance variable Hi-C features', hic.var["d_highly_variable"].sum())

    d_prior = scglue.graph.compose_multigraph(d_prior, d_prior.reverse())
    for item in itertools.chain(hic.var_names, rna.var_names):
        d_prior.add_edge(item, item, weight=1.0, type="self-loop")
    nx.set_edge_attributes(d_prior, 1, "sign")

    d_prior = d_prior.subgraph(hvg_reachable)

    dcq_prior = scglue.graph.compose_multigraph(o_prior, pchic_graph)
    dcq_prior = scglue.graph.compose_multigraph(dcq_prior, dcq_prior.reverse())
    hvg_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)

    hic.var["dcq_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print(hic.var["dcq_highly_variable"].sum())

    for item in dcq_prior.nodes:
        if dcq_prior.has_edge(item, item):
            continue
        else:
            dcq_prior.add_edge(item, item, weight=1.0, type="self-loop")
 
    nx.set_edge_attributes(dcq_prior, 1, "sign")

    dcq_prior = dcq_prior.subgraph(hvg_reachable)
    edge_count = dcq_prior.number_of_edges()

    for i, e in enumerate(dcq_prior.edges(data=True)):
        if i < 50:
            print(e)
        else:
            break

    chrom_attr = {}
    strata_attr = {}
    pos_attr = {}
    type_attr = {}
    test_i = 0
    for n in tqdm(dcq_prior.nodes):
        if test_i < 10:
            print(n)
            test_i += 1
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

    print(hic.var.head())
    print(rna.var.head())

    if use_ice:
        suffix = f'ice_{loop_q}'
    elif use_raw_pseudobulk:
        suffix = f'raw_{loop_q}'
    else:
        suffix = f'deeploop_{loop_q}'
    if use_2d_rep:
        suffix += '_2d'
    print(suffix)
    os.makedirs(os.path.join(out_dir, 'rna'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'hic'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'graphs'), exist_ok=True)

    hic.write(f"{out_dir}/hic/hic_{resolution}_{suffix}.h5ad", compression="gzip")
    rna.write(f"{out_dir}/rna/rna_{resolution}_{suffix}.h5ad", compression="gzip")

    nx.write_graphml(overlap_graph, f"{out_dir}/graphs/overlap_{resolution}_{suffix}.graphml.gz")
    nx.write_graphml(dist_graph, f"{out_dir}/graphs/dist_{resolution}_{suffix}.graphml.gz")

    nx.set_node_attributes(pchic_graph, chrom_attr, "chrom")
    nx.set_node_attributes(pchic_graph, pos_attr, "chrom_pos")
    nx.set_node_attributes(pchic_graph, type_attr, "feature_type")
    nx.set_node_attributes(pchic_graph, strata_attr, "strata")
    nx.write_graphml(pchic_graph, f"{out_dir}/graphs/hic_{resolution}_{suffix}.graphml.gz")

    nx.write_graphml(o_prior, f"{out_dir}/graphs/o_prior_{resolution}_{suffix}.graphml.gz", named_key_ids=True)
    nx.write_graphml(d_prior, f"{out_dir}/graphs/d_prior_{resolution}_{suffix}.graphml.gz", named_key_ids=True)
    nx.set_node_attributes(dcq_prior, chrom_attr, "chrom")
    nx.set_node_attributes(dcq_prior, pos_attr, "chrom_pos")
    nx.set_node_attributes(dcq_prior, type_attr, "feature_type")
    nx.set_node_attributes(dcq_prior, strata_attr, "strata")
    nx.write_graphml(dcq_prior, f"{out_dir}/graphs/dcq_prior_{resolution}_{suffix}.graphml.gz", edge_id_from_attribute='edge_id', named_key_ids=True)