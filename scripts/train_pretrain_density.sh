#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_H5="${TRAIN_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/ED_h5/train/*.h5}"
VAL_H5="${VAL_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/ED_h5/val/*.h5}"
OUT_DIR="${OUT_DIR:-}"
INIT_CKPT="${INIT_CKPT:-}"
GEOMETRIC_OUT_DIR="${GEOMETRIC_OUT_DIR:-}"
RESTART="${RESTART:-}"
ENCODER_TYPE="${ENCODER_TYPE:-}"

if [[ -z "${TRAIN_H5}" ]]; then
  echo "[error] TRAIN_H5 is required."
  exit 1
fi

read -r -a train_h5_arr <<<"${TRAIN_H5}"
if [[ "${#train_h5_arr[@]}" -eq 0 ]]; then
  echo "[error] TRAIN_H5 parsed empty."
  exit 1
fi

if [[ -z "${ENCODER_TYPE}" ]]; then
  case "${INIT_CKPT}" in
    *gotennet_l*) ENCODER_TYPE="gotennet_l" ;;
    *reacformer_se3*) ENCODER_TYPE="reacformer_se3" ;;
    *reacformer_so2*) ENCODER_TYPE="reacformer_so2" ;;
    *) ENCODER_TYPE="single_mol" ;;
  esac
fi

if [[ -z "${GEOMETRIC_OUT_DIR}" ]]; then
  GEOMETRIC_OUT_DIR="runs/${ENCODER_TYPE}_geometric"
fi

if [[ -z "${INIT_CKPT}" ]]; then
  if [[ -f "${GEOMETRIC_OUT_DIR}/best.pt" ]]; then
    INIT_CKPT="${GEOMETRIC_OUT_DIR}/best.pt"
  elif [[ -f "${GEOMETRIC_OUT_DIR}/latest.pt" ]]; then
    INIT_CKPT="${GEOMETRIC_OUT_DIR}/latest.pt"
  else
    echo "[error] Cannot find stage1 checkpoint under ${GEOMETRIC_OUT_DIR}."
    echo "Set INIT_CKPT explicitly, or ensure ${GEOMETRIC_OUT_DIR}/best.pt (or latest.pt) exists."
    exit 1
  fi
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="runs/${ENCODER_TYPE}_density"
fi

cmd=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m uni_react.train_pretrain_density
  --train_h5 "${train_h5_arr[@]}"
  --encoder_type "${ENCODER_TYPE}"
  --out_dir "${OUT_DIR}"
  --epochs 20
  --batch_size 16
  --num_workers 16
  --num_query_points 2048
  --descriptor_lr 2e-5
  --head_lr 2e-4
  --weight_decay 1e-2
  --grad_clip 1.0
  --lr_scheduler cosine
  --emb_dim 256
  --inv_layer 2
  --se3_layer 4
  --heads 8
  --atom_vocab_size 128
  --cutoff 5.0
  --num_kernel 128
  --point_hidden_dim 128
  --cond_hidden_dim 64
  --head_hidden_dim 512
  --radial_sigma 1.5
  --save_every 5
)

if [[ -n "${VAL_H5}" ]]; then
  read -r -a val_h5_arr <<<"${VAL_H5}"
  if [[ "${#val_h5_arr[@]}" -gt 0 ]]; then
    cmd+=(--val_h5 "${val_h5_arr[@]}")
  fi
fi

if [[ -n "${INIT_CKPT}" ]]; then
  if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "[error] INIT_CKPT not found: ${INIT_CKPT}"
    exit 1
  fi
  cmd+=(--init_ckpt "${INIT_CKPT}")
fi

if [[ -n "${RESTART}" ]]; then
  if [[ ! -f "${RESTART}" ]]; then
    echo "[error] RESTART checkpoint not found: ${RESTART}"
    exit 1
  fi
  cmd+=(--restart "${RESTART}")
fi

"${cmd[@]}"
