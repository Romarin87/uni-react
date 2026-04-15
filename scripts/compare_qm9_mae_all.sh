#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 直接比较 scratch vs stage1 vs stage2(CDFT) vs stage3
# 其余参数可透传，例如：
#   bash scripts/compare_qm9_mae_all.sh --output_json runs/qm9_stage123_compare.json

python -m uni_react.tools.compare_qm9_mae --compare_geometric_cdft3 "$@"
