#!/usr/bin/env bash

NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

has_arg() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

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

run_torch_module() {
  local module="$1"
  shift
  local args=("$@")
  if [[ -n "${OUT_DIR:-}" ]] && ! has_arg "--out_dir" "${args[@]}"; then
    args+=(--out_dir "${OUT_DIR}")
  fi
  echo "[INFO] Launch config: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
  exec torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m "${module}" \
    "${args[@]}"
}
