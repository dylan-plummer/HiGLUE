#!/usr/bin/env bash
# HiCorr + DeepLoop2 on pseudobulks of single-cell Hi-C / Dip-C cells (one pool per group of cells), as used for
# the islet cell-type pseudobulks of the HiGLUE paper. See ../SKILL.md for inputs, outputs and choices.
#
# Steps, per group in <out>/<group>/:
#   pools      make_pools.py: cells.tsv + cells.md5 per group from the cell table
#   hicorr     merge_pool.py: per-cell fragment pairs summed per anchor pair (end_loop.cis / .trans), then HiCorr:
#              blacklist, <= 2 Mb cis, trans-based anchor visibility, distance x anchor-length expected
#              -> split/anchor_2_anchor.loop.chrN (anchor1 anchor2 observed expected)        [CPU, --jobs groups at once]
#   deeploop2  DeepLoop2 (dense U-Net, loopdenoise_target checkpoints) with the checkpoint whose training depth is
#              closest (log scale) to the pool's cis reads <= 2 Mb -> deeploop2/chrN.denoised.anchor.to.anchor   [GPU]
#   cooler     to_cooler.py: <group>.hicorr.cool (count = obs, ratio = obs/exp), <group>.deeploop2.cool
#   loops      call_loops.py: <group>.loops.bedpe (the --n_pixels strongest DeepLoop2 pixels, 20 kb - 2 Mb, clustered;
#              one loop per cluster at its centre of mass, see call_loops.py; correct the 35M/49M/70M/98M
#              checkpoints' one-anchor output offset first)
#   heatmaps   plot_heatmaps.py, always redrawn (seconds): <group>/<group>.heatmaps.png (HiCorr | DeepLoop2 + loops)
#              and <out>/heatmaps.deeploop2.png (all groups, same regions, shared scale); regions in
#              <out>/heatmap_regions.tsv (picked from the groups' loops once, then reused)
# Every step is skipped when its output is up to date for the group's current cells (cells.md5) and checkpoint, so
# rerunning after adding cells or groups only redoes what changed.
#
# Usage:
#   run_pipeline.sh --cells cells.tsv --out /path/to/pools [options]
#   run_pipeline.sh --out /path/to/pools [options]          # rerun / resume: pools already made
# Options:
#   --genome hg19|mm10      HiCorr DpnII reference (default hg19)
#   --groups "A B"          only these groups (original names from the cell table)
#   --frag_dirs "d1 d2"     look up frag_loop.<cell>.cis.filter / .trans here when the table has no cis/trans columns
#   --min_cells N           skip smaller groups (default 1)
#   --jobs N                HiCorr groups at once (default 3; each needs ~6 cores and up to ~30 GB RAM for deep pools)
#   --threads N             reader processes per merge (default 6)
#   --gpu ID                GPU for DeepLoop2 (sets CUDA_VISIBLE_DEVICES; default: as set in the environment)
#   --checkpoint FILE       a DeepLoop2 checkpoint for every group instead of the depth-matched one
#   --chroms "chr21 chr22"  DeepLoop2 / coolers on these chromosomes only (tests)
#   --steps "..."           subset of: pools hicorr deeploop2 cooler loops heatmaps (default: all)
#   --n_pixels N            strongest DeepLoop2 pixels clustered into loops, per group (default 500000)
#   --regions "chr:s-e ..." heatmap regions (default: 4 x 1.5 Mb picked from the loops, kept in heatmap_regions.tsv)
#   --compact               after the coolers, delete the large text intermediates (split/, deeploop2/, end_loop.*;
#                           ~30 GB per group); a later DeepLoop2 rerun then redoes HiCorr first
# Environment: PYTHON (default python) needs numpy pandas scipy torch tqdm cooler; perl for HiCorr.
#   HICORR_LIB, HICORR_REF, DL2_DIR override the lab's HiCorr scripts, references and DeepLoop2 locations.
#   HICORR_ENZYME (default DpnII) names the anchor files: $HICORR_REF/<genome>_<enzyme>_anchors_avg.bed and
#   _frag_2_anchor (e.g. HICORR_ENZYME=GATC_CATG with a reference directory for DpnII + NlaIII, snm3C-seq).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python}
LIB=${HICORR_LIB:-/mnt/jinstore/Archive01/xww/for_xiaoxiao/HiCorr/src}
REF=${HICORR_REF:-/mnt/jinstore/JinLab03/xxl1432/Reference/HiC/xww_HiCorr/references}
DL2=${DL2_DIR:-/mnt/jinstore/Archive01/kxt439/DeepLoop2.0}
CKPTS=$DL2/ckpts/loopdenoise_target
ENZYME=${HICORR_ENZYME:-DpnII}
# HiCorr's merge_sorted_anchor_loop.pl walks its inputs (pool contacts, reference full matrix) in Perl hash order,
# and for pixels with reads the distance of the input read last wins; the two inputs use different distance
# conventions, so with Perl's per-run hash randomisation ~2% of expected values moved by ~0.7% between runs. A fixed
# hash order makes reruns reproducible.
export PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0

cells=""; out=""; genome=hg19; groups=""; frag_dirs=""; min_cells=1; JOBS=3; THREADS=6; ckpt=""; chroms=""
steps="pools hicorr deeploop2 cooler loops heatmaps"; n_pixels=500000; compact=0; regions=""
die() { echo "!! $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
    case $1 in
        --cells) cells=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --genome) genome=$2; shift 2 ;;
        --groups) groups=$2; shift 2 ;;
        --frag_dirs) frag_dirs=$2; shift 2 ;;
        --min_cells) min_cells=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --threads) THREADS=$2; shift 2 ;;
        --gpu) export CUDA_VISIBLE_DEVICES=$2; shift 2 ;;
        --checkpoint) ckpt=$(readlink -f "$2"); shift 2 ;;
        --chroms) chroms=$2; shift 2 ;;
        --steps) steps=$2; shift 2 ;;
        --n_pixels) n_pixels=$2; shift 2 ;;
        --regions) regions=$2; shift 2 ;;
        --compact) compact=1; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
[[ -n $out ]] || die "--out is required"
has() { [[ " $steps " == *" $1 "* ]]; }

# ---- checks before any long step
anchorbed=$REF/${genome}_${ENZYME}_anchors_avg.bed
frag2anchor=$REF/${genome}_${ENZYME}_frag_2_anchor
blacklist=$REF/${genome}_5kb_anchors_blacklist
for f in "$anchorbed" "$frag2anchor" "$blacklist" "$REF/${genome}_anchor_length.groups" "$REF/${genome}.dist.401.group" \
         "$REF/${genome}_full_filtered_matrix" "$REF/${genome}.trans.possible.pairs"; do
    [[ -e $f ]] || die "HiCorr reference missing for --genome $genome: $f"
done
for f in get_dist.py remove.blacklist.py calculate_vis.py split_chromo.py merge_sorted_anchor_loop.pl \
         get_group_statistics.pl get_trans.avg_by_len.pl get_corr_factor_by_len.pl correct.trans.reads.by.corr.pl \
         get_loop_lambda.pl add.vis.to.cis.2M.pl; do
    [[ -e $LIB/$f ]] || die "HiCorr script missing: $LIB/$f"
done
command -v perl > /dev/null || die "perl not found"
$PY -c "import numpy, pandas, scipy, cooler" 2> /dev/null || die "$PY lacks numpy/pandas/scipy/cooler (set PYTHON=)"
if has heatmaps; then $PY -c "import matplotlib" 2> /dev/null || die "$PY lacks matplotlib, needed by the heatmaps step"; fi
if has deeploop2; then
    $PY -c "import torch, tqdm" 2> /dev/null || die "$PY lacks torch/tqdm, needed by DeepLoop2"
    [[ -n $ckpt ]] || compgen -G "$CKPTS/dl2_ld_1.0_*M_300k.pt" > /dev/null || die "no checkpoints in $CKPTS"
    [[ -z $ckpt || -f $ckpt ]] || die "--checkpoint $ckpt not found"
    $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || echo "WARNING: no GPU visible to torch; DeepLoop2 runs on the CPU (many hours per group)"
fi
[[ $genome == hg19 ]] || echo "NOTE: DeepLoop2 was trained on human data; on $genome its output is unvalidated"

mkdir -p "$out"
out=$(cd "$out" && pwd)
echo "=== HiCorr + DeepLoop2  $(date -Is)  out=$out genome=$genome steps='$steps'"

# ---- 1. pools
if has pools; then
    [[ -n $cells ]] || [[ -f $out/groups.tsv ]] || die "--cells is required on the first run"
    if [[ -n $cells ]]; then
        args=(--cells "$cells" --out "$out" --min_cells "$min_cells")
        if [[ -n $groups ]]; then read -r -a g <<< "$groups"; args+=(--groups "${g[@]}"); fi
        if [[ -n $frag_dirs ]]; then read -r -a fd <<< "$frag_dirs"; args+=(--frag_dirs "${fd[@]}"); fi
        $PY "$here/make_pools.py" "${args[@]}"
    fi
fi
[[ -f $out/groups.tsv ]] || die "no $out/groups.tsv: run the pools step (--cells) first"
dirs=()
while IFS=$'\t' read -r g d n; do
    if [[ -n $groups ]]; then
        hit=0; for x in $groups; do [[ $x == "$g" ]] && hit=1; done
        (( hit )) || continue
    fi
    dirs+=("$d")
done < <(tail -n +2 "$out/groups.tsv")
(( ${#dirs[@]} )) || die "no groups to run"
echo "groups: ${dirs[*]}"

# ---- shared code: HiCorr's python 2 helpers as python 3 copies; DeepLoop2 inference and per-chromosome anchors
src=$out/src
mkdir -p "$src"
for f in get_dist.py remove.blacklist.py calculate_vis.py split_chromo.py; do
    sed -E 's/^([[:space:]]*)print (.*)$/\1print(\2)/' "$LIB/$f" > "$src/$f"
done
if has deeploop2; then
    # copied, so later edits in the DeepLoop2 tree do not change results made here
    [[ -f $src/predict.py ]] || cp "$DL2/predict/dl2_predict/predict.py" "$DL2/predict/dl2_predict/dense_unet.py" "$src/"
    dl2_anchors=$src/anchors_$genome
    if [[ ! -d $dl2_anchors ]]; then
        mkdir -p "$dl2_anchors.tmp"
        awk 'BEGIN{OFS="\t"} {print $1, $2, $3, $4 > (dir "/" $1 ".bed")}' dir="$dl2_anchors.tmp" "$anchorbed"
        mv "$dl2_anchors.tmp" "$dl2_anchors"
    fi
fi

key_of() { cat "$out/$1/cells.md5"; }
guard() { [[ -f $out/$1/cells.md5 ]] || die "$out/$1 has no cells.md5: not a pipeline directory"; }

hicorr() {
    local t=$1 d=$out/$1 key
    guard "$t"; key=$(key_of "$t")
    if [[ -f $d/.hicorr_done && $(cat "$d/.hicorr_done") == "$key" && -d $d/split ]]; then
        echo "[$t] HiCorr up to date"; return
    fi
    echo "[$t] HiCorr start $(date -Is) ($(( $(wc -l < "$d/cells.tsv") - 1 )) cells)"
    rm -rf "$d/split" "$d/deeploop2" "$d"/.hicorr_done "$d"/.dl2_done "$d"/.cool_done "$d"/.loops_done "$d"/end_loop.* \
        "$d"/*.cool "$d"/*.loops.bedpe
    $PY "$here/merge_pool.py" --group_dir "$d" --frag2anchor "$frag2anchor" --threads "$THREADS"
    (
        cd "$d"
        cat end_loop.cis | python3 "$src/get_dist.py" "$anchorbed" | python3 "$src/remove.blacklist.py" "$blacklist" > end_loop.cis.rmbl
        cat end_loop.trans | python3 "$src/remove.blacklist.py" "$blacklist" > end_loop.rmbl.trans
        awk '{if($4<=2000000) print $0}' end_loop.cis.rmbl > end_loop.2M.rmbl
        {
            echo "cis_reads $(awk '{s+=$3}END{print s/2}' end_loop.cis.rmbl)"
            echo "cis_2M_reads $(awk '{s+=$3}END{print s/2}' end_loop.2M.rmbl)"
            echo "trans_reads $(awk '{s+=$3}END{print s/2}' end_loop.rmbl.trans)"
        } > summary
        $LIB/merge_sorted_anchor_loop.pl $REF/${genome}_full_filtered_matrix end_loop.2M.rmbl > end_loop.full
        cat end_loop.full | $LIB/get_group_statistics.pl - $anchorbed $REF/${genome}_anchor_length.groups $REF/${genome}.dist.401.group > integrated.dist.len.stat
        awk '{if($4>2000000) print $0}' end_loop.cis.rmbl | cat - end_loop.rmbl.trans > merged.2M.trans
        $LIB/get_trans.avg_by_len.pl merged.2M.trans $REF/${genome}_anchor_length.groups $anchorbed $REF/${genome}.trans.possible.pairs \
            | $LIB/get_corr_factor_by_len.pl - > len.factor
        $LIB/correct.trans.reads.by.corr.pl merged.2M.trans $anchorbed $REF/${genome}_anchor_length.groups len.factor \
            | python3 "$src/calculate_vis.py" > anchor.vis.list
        perl $LIB/get_loop_lambda.pl end_loop.full $anchorbed $REF/${genome}_anchor_length.groups $REF/${genome}.dist.401.group integrated.dist.len.stat \
            | $LIB/add.vis.to.cis.2M.pl - anchor.vis.list | python3 "$src/split_chromo.py" $anchorbed
        rm -f end_loop.full merged.2M.trans end_loop.cis end_loop.cis.rmbl  # large intermediates
    )
    echo "$key" > "$d/.hicorr_done"
    echo "[$t] HiCorr done $(date -Is): $(tr '\n' ' ' < "$d/summary")"
}

pick_model() {  # checkpoint whose training depth is closest (log scale) to the pool's cis reads <= 2 Mb
    [[ -n $ckpt ]] && { echo "$ckpt"; return; }
    ls "$CKPTS"/dl2_ld_1.0_*M_300k.pt | awk -v r="$1" '{
        m = $0; sub(/.*dl2_ld_1.0_/, "", m); sub(/M_300k.pt/, "", m)
        d = log(r / (m * 1e6)); if (d < 0) d = -d
        if (best == "" || d < bd) {best = $0; bd = d}
    } END {print best}'
}

deeploop2() {
    local t=$1 d=$out/$1 key reads model
    guard "$t"; key=$(key_of "$t")
    [[ -d $d/split ]] || { echo "[$t] no HiCorr output (split/), running HiCorr first"; rm -f "$d/.hicorr_done"; hicorr "$t"; }
    reads=$(awk '$1=="cis_2M_reads"{print $2}' "$d/summary")
    model=$(pick_model "$reads")
    # up to date = same cells, same checkpoint (new checkpoints change the pick) and same chromosomes
    if [[ -f $d/.dl2_done && $(sed -n 1p "$d/.dl2_done") == "$key" && $(sed -n 2p "$d/.dl2_done") == "$(basename "$model")" \
          && $(sed -n 4p "$d/.dl2_done") == "${chroms:-all}" ]]; then
        echo "[$t] DeepLoop2 up to date ($(basename "$model"))"; return
    fi
    rm -f "$d/.cool_done" "$d/.loops_done"
    echo "[$t] DeepLoop2 start $(date -Is): $reads cis reads <= 2 Mb, checkpoint $(basename "$model")"
    awk -v r="$reads" 'BEGIN{ if (r < 3.5e6) print "  WARNING: below half the smallest checkpoint (7 M): expect a smear along the diagonal" }'
    rm -rf "$d/deeploop2"; mkdir -p "$d/deeploop2"
    local f chrom
    for f in "$d"/split/anchor_2_anchor.loop.chr*; do
        chrom=${f##*.}
        [[ -z $chroms || " $chroms " == *" $chrom "* ]] || continue
        [[ -f $dl2_anchors/$chrom.bed ]] || continue
        if ! $PY "$src/predict.py" --full_matrix_dir "$d/split" --input_name "$(basename "$f")" --weights "$model" \
                --out_dir "$d/deeploop2" --anchors "$dl2_anchors" --chromosome "$chrom" \
                --val_cols obs exp --small_matrix_size 128 --step_size 64 --max_dist 768 --dummy 5 > "$d/deeploop2.$chrom.log" 2>&1; then
            die "[$t] DeepLoop2 failed on $chrom, see $d/deeploop2.$chrom.log"
        fi
    done
    printf '%s\n%s\n%s\n%s\n' "$key" "$(basename "$model")" "$reads" "${chroms:-all}" > "$d/.dl2_done"
    echo "[$t] DeepLoop2 done $(date -Is)"
}

to_cooler() {
    local t=$1 d=$out/$1 key
    guard "$t"; key=$(key_of "$t")
    if [[ -f $d/.cool_done && $(cat "$d/.cool_done") == "$key" ]]; then echo "[$t] coolers up to date"; return; fi
    [[ -d $d/split && -d $d/deeploop2 ]] || die "[$t] run the hicorr and deeploop2 steps first"
    $PY "$here/to_cooler.py" --group_dir "$d" --anchors "$anchorbed" --assembly "$genome"
    echo "$key" > "$d/.cool_done"
    if (( compact )); then
        rm -rf "$d/split" "$d/deeploop2" "$d"/end_loop.*
        echo "[$t] --compact: text intermediates removed"
    fi
}

loops() {
    local t=$1 d=$out/$1 key
    guard "$t"; key=$(key_of "$t")
    if [[ -f $d/.loops_done && $(cat "$d/.loops_done") == "$key cluster $n_pixels" ]]; then echo "[$t] loops up to date"; return; fi
    [[ -f $d/$t.deeploop2.cool ]] || die "[$t] run the cooler step first"
    $PY "$here/call_loops.py" --group_dir "$d" --n_pixels "$n_pixels"
    echo "$key cluster $n_pixels" > "$d/.loops_done"
}

# ---- 2. HiCorr: CPU only, up to --jobs groups at once
if has hicorr; then
    pids=(); names=()
    for t in "${dirs[@]}"; do
        while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do sleep 20; done
        hicorr "$t" > "$out/$t/hicorr.log" 2>&1 &
        pids+=($!); names+=("$t")
    done
    fail=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { fail=1; echo "!! [${names[$i]}] HiCorr failed, see $out/${names[$i]}/hicorr.log"; }
    done
    for t in "${dirs[@]}"; do tail -1 "$out/$t/hicorr.log"; done
    (( fail == 0 )) || exit 1
fi
# ---- 3-5. DeepLoop2 group by group on the GPU, then coolers and loops
for t in "${dirs[@]}"; do
    if has deeploop2; then deeploop2 "$t"; fi
    if has cooler; then to_cooler "$t"; fi
    if has loops; then loops "$t"; fi
done
# ---- 6. test heatmaps: every group, same regions (cheap, so always redrawn)
if has heatmaps; then
    ready=()
    for t in "${dirs[@]}"; do
        [[ -f $out/$t/$t.deeploop2.cool && -f $out/$t/$t.hicorr.cool ]] && ready+=("$t") || echo "[$t] no coolers yet, no heatmaps"
    done
    if (( ${#ready[@]} )); then
        args=(--out "$out" --groups "${ready[@]}")
        [[ -n $regions ]] && args+=(--regions "$regions")
        $PY "$here/plot_heatmaps.py" "${args[@]}" || die "heatmaps failed (outputs above are complete; rerun with --steps heatmaps)"
    fi
fi
echo "=== done $(date -Is)"
