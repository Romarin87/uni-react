#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/gotennet_l_hat/qm9.yaml}"
DATA_ROOT="${DATA_ROOT:-./data/qm9_pyg}"
OUT_ROOT="${OUT_ROOT:-runs}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
RUN_FAMILY="${RUN_FAMILY:-}"
NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

if [[ -n "${PET_NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
elif [[ -n "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#CUDA_DEVICES[@]}"
else
  NPROC_PER_NODE=8
fi

TARGETS=(
  mu
  alpha
  homo
  lumo
  gap
  r2
  zpve
  U0
  U
  H
  G
  Cv
)

infer_run_family() {
  local ckpt="$1"
  if [[ -z "${ckpt}" ]]; then
    echo "scratch"
    return
  fi
  local lowered="${ckpt,,}"
  if [[ "${lowered}" == *reaction* ]]; then
    echo "finetune_from_reaction"
  elif [[ "${lowered}" == *density* ]]; then
    echo "finetune_from_density"
  elif [[ "${lowered}" == *cdft* ]] || [[ "${lowered}" == *electronic* ]]; then
    echo "finetune_from_cdft"
  elif [[ "${lowered}" == *geometric* ]]; then
    echo "finetune_from_geometric"
  else
    echo "finetune"
  fi
}

if [[ -z "${RUN_FAMILY}" ]]; then
  RUN_FAMILY="$(infer_run_family "${PRETRAINED_CKPT}")"
fi

echo "[INFO] Launch config: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

for target in "${TARGETS[@]}"; do
  out_dir="${OUT_ROOT}/qm9_${RUN_FAMILY}_gotennet_l_hat_${target}"
  echo "==> Training GotenNet-L_HAT on QM9 target: ${target}"
  echo "    out_dir=${out_dir}"
  cmd=(
    torchrun
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    -m uni_react.train_qm9
    --config "${CONFIG}"
    --model_name gotennet_l_hat
    --data_root "${DATA_ROOT}"
    --targets "${target}"
    --out_dir "${out_dir}"
  )
  if [[ -n "${PRETRAINED_CKPT}" ]]; then
    cmd+=(--pretrained_ckpt "${PRETRAINED_CKPT}")
  fi
  "${cmd[@]}" "$@"
done
