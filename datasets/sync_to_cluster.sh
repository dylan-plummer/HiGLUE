#!/usr/bin/env bash
# Copies the dataset tree (see link_datasets.sh) to the cluster. Links to files
# outside the tree are copied as real files; links within the tree (to
# _shared/) stay links, so shared GTFs and RNA references are copied once.
#
# Usage:
#   datasets/sync_to_cluster.sh                                  # everything (~30 GB)
#   datasets/sync_to_cluster.sh --datasets islet_atlas,islet_inhouse
#   datasets/sync_to_cluster.sh --res "5kb 100kb 500kb"          # only these scools
#   datasets/sync_to_cluster.sh --dest user@host:/path/datasets --dry_run
#
# The default destination is the HiGLUE checkout on the cluster, where
# benchmark_main.sh finds the data at datasets/ (its default --data_root).

set -euo pipefail
cd "$(dirname "$0")"

DEST=${DEST:-dmp131@hpctransfer1.case.edu:/scratch/pioneer/users/dmp131/}
DATASETS=""
RES=""
DRY_RUN=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dest) DEST=$2; shift 2 ;;
        --datasets) DATASETS=$2; shift 2 ;;
        --res) RES=$2; shift 2 ;;
        --dry_run) DRY_RUN=(--dry-run); shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

FILTERS=()
if [[ -n $RES ]]; then
    for r in $RES; do FILTERS+=(--include "hic_${r}.scool"); done
    FILTERS+=(--exclude "hic_*.scool")
fi
if [[ -n $DATASETS ]]; then
    FILTERS+=(--include "_shared/***")
    IFS=',' read -r -a selected <<< "$DATASETS"
    for d in "${selected[@]}"; do FILTERS+=(--include "$d/***"); done
    FILTERS+=(--exclude "*")
fi
# scripts and notes stay in the repository
FILTERS=(--exclude "*.sh" --exclude "README.md" "${FILTERS[@]}")

rsync -a --copy-unsafe-links --info=progress2 --human-readable "${DRY_RUN[@]}" \
    "${FILTERS[@]}" ./ "$DEST"
