#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_H5="${TRAIN_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/GDB13/gdb13_chno_hdf5/gdb13_chno_hdf5_train/*.h5}"
VAL_H5="${VAL_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/GDB13/gdb13_chno_hdf5/gdb13_chno_hdf5_val/*.h5}"
ENCODER_TYPE="${ENCODER_TYPE:-single_mol}"
OUT_DIR="${OUT_DIR:-runs/${ENCODER_TYPE}_geometric}"
RESTART="${RESTART:-}"

if [[ -z "${TRAIN_H5}" ]]; then
  echo "[error] TRAIN_H5 is required."
  echo "Example: TRAIN_H5='/path/to/train1.h5 /path/to/train2.h5' bash scripts/train_pretrain_geometric.sh"
  exit 1
fi

read -r -a train_h5_arr <<<"${TRAIN_H5}"
if [[ "${#train_h5_arr[@]}" -eq 0 ]]; then
  echo "[error] TRAIN_H5 parsed empty."
  exit 1
fi

cmd=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m uni_react.train_pretrain_geometric
  --train_h5 "${train_h5_arr[@]}"
  --encoder_type "${ENCODER_TYPE}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${VAL_H5}" ]]; then
  read -r -a val_h5_arr <<<"${VAL_H5}"
  cmd+=(--val_h5 "${val_h5_arr[@]}")
fi

if [[ -n "${RESTART}" ]]; then
  cmd+=(--restart "${RESTART}")
fi

"${cmd[@]}"
