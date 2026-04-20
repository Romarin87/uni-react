#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/gotennet_l/qm9.yaml}"
DATA_ROOT="${DATA_ROOT:-./data/qm9_pyg}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUT_ROOT="${OUT_ROOT:-runs}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
RUN_FAMILY="${RUN_FAMILY:-}"

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

for target in "${TARGETS[@]}"; do
  out_dir="${OUT_ROOT}/qm9_${RUN_FAMILY}_gotennet_l_${target}"
  echo "==> Training GotenNet-L on QM9 target: ${target}"
  echo "    out_dir=${out_dir}"
  cmd=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    -m uni_react.train_finetune_qm9
    --config "${CONFIG}"
    --model_name gotennet_l
    --data_root "${DATA_ROOT}"
    --targets "${target}"
    --out_dir "${out_dir}"
  )
  if [[ -n "${PRETRAINED_CKPT}" ]]; then
    cmd+=(--pretrained_ckpt "${PRETRAINED_CKPT}")
  fi
  "${cmd[@]}" "$@"
done
