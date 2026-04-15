#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------
# 可修改参数（默认值）
# ---------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DATA_ROOT="${DATA_ROOT:-qm9_pyg}"
SPLIT="${SPLIT:-egnn}"                # egnn | dimenet
FORCE_RELOAD="${FORCE_RELOAD:-0}"     # 1=强制重新处理/下载 PyG QM9
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
PRETRAINED_STRICT="${PRETRAINED_STRICT:-0}" # 1=加 --pretrained_strict
ENCODER_TYPE="${ENCODER_TYPE:-single_mol}"
RESTART="${RESTART:-}"
RESTART_IGNORE_CONFIG="${RESTART_IGNORE_CONFIG:-0}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"

HEAD_LR="${HEAD_LR:-auto}"                # auto=按 EGNN target 规则：gap/homo/lumo 用 1e-3，其余 5e-4
BACKBONE_LR="${BACKBONE_LR:-auto}"        # auto=从 HEAD_LR 按统一比例缩放
EGNN_BASE_LR="${EGNN_BASE_LR:-5e-4}"
EGNN_FAST_LR="${EGNN_FAST_LR:-1e-3}"
EGNN_FAST_TARGETS="${EGNN_FAST_TARGETS:-gap homo lumo}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"

EMB_DIM="${EMB_DIM:-256}"
INV_LAYER="${INV_LAYER:-2}"
SE3_LAYER="${SE3_LAYER:-4}"
HEADS="${HEADS:-8}"
ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-128}"
CUTOFF="${CUTOFF:-5.0}"
NUM_KERNEL="${NUM_KERNEL:-128}"
PATH_DROPOUT="${PATH_DROPOUT:-0.0}"
ACTIVATION_DROPOUT="${ACTIVATION_DROPOUT:-0.0}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.05}"

# r2 特例参数（仅在 target=r2 时生效；留空表示沿用通用参数）
R2_BATCH_SIZE="${R2_BATCH_SIZE:-}"
R2_NUM_WORKERS="${R2_NUM_WORKERS:-}"
R2_EPOCHS="${R2_EPOCHS:-}"
R2_SAVE_EVERY="${R2_SAVE_EVERY:-}"
R2_BACKBONE_LR="${R2_BACKBONE_LR:-}"
R2_HEAD_LR="${R2_HEAD_LR:-}"
R2_WEIGHT_DECAY="${R2_WEIGHT_DECAY:-}"
R2_CUTOFF="${R2_CUTOFF:-}"
R2_HEAD_DROPOUT="${R2_HEAD_DROPOUT:-}"

NO_CENTER_COORDS="${NO_CENTER_COORDS:-0}" # 1=加 --no_center_coords
OUT_ROOT="${OUT_ROOT:-runs}"
OUT_PREFIX="${OUT_PREFIX:-qm9}"

infer_run_family() {
  local ckpt="${1:-}"
  local ckpt_lower=""
  if [[ -z "${ckpt}" ]]; then
    echo "scratch"
    return
  fi
  ckpt_lower="$(echo "${ckpt}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${ckpt_lower}" == *reaction* ]]; then
    echo "pretrain_reaction"
    return
  fi
  if [[ "${ckpt_lower}" == *density* ]]; then
    echo "pretrain_density"
    return
  fi
  if [[ "${ckpt_lower}" == *cdft* || "${ckpt_lower}" == *electronic* ]]; then
    echo "pretrain_cdft"
    return
  fi
  if [[ "${ckpt_lower}" == *geometric* ]]; then
    echo "pretrain_geometric"
    return
  fi

  local source=""
  local base
  base="$(basename "${ckpt}")"
  if [[ "${base}" == "best.pt" || "${base}" == "latest.pt" || "${base}" == epoch_*.pt ]]; then
    source="$(basename "$(dirname "${ckpt}")")"
  else
    source="${base%.pt}"
  fi
  source="$(echo "${source}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  if [[ -z "${source}" ]]; then
    source="custom"
  fi
  echo "pretrain_${source}"
}

infer_prefix_from_restart() {
  local restart_path="${1:-}"
  local run_dir=""
  local run_name=""
  local target_lower="${2:-}"
  if [[ -z "${restart_path}" ]]; then
    return
  fi
  run_dir="$(dirname "${restart_path}")"
  run_name="$(basename "${run_dir}")"
  if [[ -n "${target_lower}" && "${run_name}" == *"_${target_lower}" ]]; then
    echo "${run_name%_${target_lower}}"
  else
    echo "${run_name}"
  fi
}

if [[ -n "${TARGETS_OVERRIDE:-}" ]]; then
  read -r -a TARGETS <<<"${TARGETS_OVERRIDE}"
else
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
fi

RUN_FAMILY="$(infer_run_family "${PRETRAINED_CKPT}")"
if [[ "${OUT_PREFIX}" == "qm9" && -n "${RESTART}" && "${#TARGETS[@]}" -eq 1 ]]; then
  target_override_lower="$(echo "${TARGETS[0]}" | tr '[:upper:]' '[:lower:]')"
  restart_prefix="$(infer_prefix_from_restart "${RESTART}" "${target_override_lower}")"
  if [[ -n "${restart_prefix}" ]]; then
    OUT_PREFIX="${restart_prefix}"
  fi
fi
if [[ "${OUT_PREFIX}" == "qm9" ]]; then
  OUT_PREFIX="qm9_${RUN_FAMILY}_${ENCODER_TYPE}_${SPLIT}"
fi

if [[ -n "${RESTART}" && "${#TARGETS[@]}" -ne 1 ]]; then
  echo "ERROR: RESTART is only supported when running a single QM9 target." >&2
  echo "Set TARGETS_OVERRIDE to one target, or leave RESTART empty for batch runs." >&2
  exit 2
fi

echo "Start sequential finetune for ${#TARGETS[@]} QM9 targets..."
echo "data_root=${DATA_ROOT}, split=${SPLIT}, nproc=${NPROC_PER_NODE}"
echo "encoder_type=${ENCODER_TYPE}"
echo "run_family=${RUN_FAMILY}"
echo "batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}, epochs=${EPOCHS}"
echo "backbone_lr=${BACKBONE_LR}, head_lr=${HEAD_LR}, wd=${WEIGHT_DECAY}"
echo "egnn_lr(base/fast)=${EGNN_BASE_LR}/${EGNN_FAST_LR}, fast_targets=[${EGNN_FAST_TARGETS}]"
echo "backbone_lr_scale=${BACKBONE_LR_SCALE}"
echo "dropout(path/act/attn/head)=${PATH_DROPOUT}/${ACTIVATION_DROPOUT}/${ATTN_DROPOUT}/${HEAD_DROPOUT}"

for target in "${TARGETS[@]}"; do
  target_lower="$(echo "${target}" | tr '[:upper:]' '[:lower:]')"
  out_dir="${OUT_ROOT}/${OUT_PREFIX}_${target_lower}"

  target_batch_size="${BATCH_SIZE}"
  target_num_workers="${NUM_WORKERS}"
  target_epochs="${EPOCHS}"
  target_save_every="${SAVE_EVERY}"
  target_weight_decay="${WEIGHT_DECAY}"
  target_cutoff="${CUTOFF}"
  target_head_dropout="${HEAD_DROPOUT}"
  target_head_lr_cfg="${HEAD_LR}"
  target_backbone_lr_cfg="${BACKBONE_LR}"

  if [[ "${target_lower}" == "r2" ]]; then
    [[ -n "${R2_BATCH_SIZE}" ]] && target_batch_size="${R2_BATCH_SIZE}"
    [[ -n "${R2_NUM_WORKERS}" ]] && target_num_workers="${R2_NUM_WORKERS}"
    [[ -n "${R2_EPOCHS}" ]] && target_epochs="${R2_EPOCHS}"
    [[ -n "${R2_SAVE_EVERY}" ]] && target_save_every="${R2_SAVE_EVERY}"
    [[ -n "${R2_WEIGHT_DECAY}" ]] && target_weight_decay="${R2_WEIGHT_DECAY}"
    [[ -n "${R2_CUTOFF}" ]] && target_cutoff="${R2_CUTOFF}"
    [[ -n "${R2_HEAD_DROPOUT}" ]] && target_head_dropout="${R2_HEAD_DROPOUT}"
    [[ -n "${R2_HEAD_LR}" ]] && target_head_lr_cfg="${R2_HEAD_LR}"
    [[ -n "${R2_BACKBONE_LR}" ]] && target_backbone_lr_cfg="${R2_BACKBONE_LR}"
  fi

  echo "============================================================"
  echo "Target: ${target}"
  echo "Out dir: ${out_dir}"
  date

  effective_head_lr="${target_head_lr_cfg}"
  if [[ "${effective_head_lr}" == "auto" ]]; then
    effective_head_lr="${EGNN_BASE_LR}"
    for fast_target in ${EGNN_FAST_TARGETS}; do
      if [[ "${target}" == "${fast_target}" ]]; then
        effective_head_lr="${EGNN_FAST_LR}"
        break
      fi
    done
  fi

  effective_backbone_lr="${target_backbone_lr_cfg}"
  if [[ "${effective_backbone_lr}" == "auto" ]]; then
    effective_backbone_lr="$(
      awk -v a="${effective_head_lr}" -v s="${BACKBONE_LR_SCALE}" 'BEGIN { printf "%.10g", a * s }'
    )"
  fi

  echo "Effective LR: backbone=${effective_backbone_lr}, head=${effective_head_lr}"
  echo "Effective cfg: bs=${target_batch_size}, workers=${target_num_workers}, epochs=${target_epochs}, wd=${target_weight_decay}, cutoff=${target_cutoff}, head_dropout=${target_head_dropout}"

  cmd=(
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m uni_react.train_finetune_qm9
    --data_root "${DATA_ROOT}"
    --split "${SPLIT}"
    --targets "${target}"
    --encoder_type "${ENCODER_TYPE}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --batch_size "${target_batch_size}"
    --num_workers "${target_num_workers}"
    --epochs "${target_epochs}"
    --save_every "${target_save_every}"
    --backbone_lr "${effective_backbone_lr}"
    --head_lr "${effective_head_lr}"
    --weight_decay "${target_weight_decay}"
    --grad_clip "${GRAD_CLIP}"
    --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}"
    --emb_dim "${EMB_DIM}"
    --inv_layer "${INV_LAYER}"
    --se3_layer "${SE3_LAYER}"
    --heads "${HEADS}"
    --atom_vocab_size "${ATOM_VOCAB_SIZE}"
    --cutoff "${target_cutoff}"
    --num_kernel "${NUM_KERNEL}"
    --path_dropout "${PATH_DROPOUT}"
    --activation_dropout "${ACTIVATION_DROPOUT}"
    --attn_dropout "${ATTN_DROPOUT}"
    --head_hidden_dim "${HEAD_HIDDEN_DIM}"
    --head_dropout "${target_head_dropout}"
    --out_dir "${out_dir}"
  )

  if [[ -n "${PRETRAINED_CKPT}" ]]; then
    cmd+=(--pretrained_ckpt "${PRETRAINED_CKPT}")
  fi
  if [[ "${PRETRAINED_STRICT}" == "1" ]]; then
    cmd+=(--pretrained_strict)
  fi
  if [[ "${NO_CENTER_COORDS}" == "1" ]]; then
    cmd+=(--no_center_coords)
  fi
  if [[ "${FORCE_RELOAD}" == "1" ]]; then
    cmd+=(--force_reload)
  fi
  if [[ -n "${RESTART}" ]]; then
    cmd+=(--restart "${RESTART}")
  fi
  if [[ "${RESTART_IGNORE_CONFIG}" == "1" ]]; then
    cmd+=(--restart_ignore_config)
  fi

  "${cmd[@]}"
done

echo "All targets finished."
