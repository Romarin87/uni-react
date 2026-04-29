#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/geometric/train_gotennet_l.sh}"
LOW="${LOW:-1}"
HIGH="${HIGH:-512}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-180}"
OUT_ROOT="${OUT_ROOT:-runs/batch_probe}"
KEEP_RUNS="${KEEP_RUNS:-0}"
PROBE_SMOKE_ARGS="${PROBE_SMOKE_ARGS:-1}"
SMOKE_H5_FILE_LIMIT="${SMOKE_H5_FILE_LIMIT:-1}"
SMOKE_TRAIN_BATCH_LIMIT="${SMOKE_TRAIN_BATCH_LIMIT:-100}"
SMOKE_VAL_BATCH_LIMIT="${SMOKE_VAL_BATCH_LIMIT:-1}"

mkdir -p "${OUT_ROOT}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "[ERROR] GNU timeout is required. Run this on the Linux training node." >&2
  exit 2
fi

is_oom_log() {
  local log_file="$1"
  grep -Eiq "CUDA out of memory|torch\\.OutOfMemoryError|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory" "${log_file}"
}

run_probe() {
  local batch_size="$1"
  shift
  local run_dir="${OUT_ROOT}/bs_${batch_size}"
  local log_file="${OUT_ROOT}/bs_${batch_size}.log"

  rm -rf "${run_dir}"
  echo "[PROBE] batch_size=${batch_size} timeout=${PROBE_TIMEOUT}s"

  local probe_args=(
    --batch_size "${batch_size}"
    --epochs 1
    --save_every 999999
    --save_every_steps 0
    --log_interval 1
    --out_dir "${run_dir}"
  )
  if [[ "${PROBE_SMOKE_ARGS}" == "1" ]]; then
    probe_args+=(
      --smoke_h5_file_limit "${SMOKE_H5_FILE_LIMIT}"
      --smoke_train_batch_limit "${SMOKE_TRAIN_BATCH_LIMIT}"
      --smoke_val_batch_limit "${SMOKE_VAL_BATCH_LIMIT}"
    )
  fi

  set +e
  timeout "${PROBE_TIMEOUT}" \
    bash "${TRAIN_SCRIPT}" "${probe_args[@]}" "$@" \
      >"${log_file}" 2>&1
  local code=$?
  set -e

  if is_oom_log "${log_file}"; then
    echo "[OOM] batch_size=${batch_size} log=${log_file}"
    [[ "${KEEP_RUNS}" == "1" ]] || rm -rf "${run_dir}"
    return 1
  fi

  if [[ "${code}" == "0" || "${code}" == "124" ]]; then
    if [[ "${code}" == "124" ]]; then
      echo "[OK] batch_size=${batch_size} survived timeout log=${log_file}"
    else
      echo "[OK] batch_size=${batch_size} finished log=${log_file}"
    fi
    [[ "${KEEP_RUNS}" == "1" ]] || rm -rf "${run_dir}"
    return 0
  fi

  echo "[ERROR] batch_size=${batch_size} failed with exit_code=${code}, not recognized as OOM. See ${log_file}" >&2
  tail -80 "${log_file}" >&2 || true
  exit "${code}"
}

best=0
lo="${LOW}"
hi="${HIGH}"

while (( lo <= hi )); do
  mid=$(( (lo + hi) / 2 ))
  if run_probe "${mid}" "$@"; then
    best="${mid}"
    lo=$(( mid + 1 ))
  else
    hi=$(( mid - 1 ))
  fi
done

echo "[RESULT] max_batch_size=${best}"
echo "[RESULT] logs=${OUT_ROOT}"
