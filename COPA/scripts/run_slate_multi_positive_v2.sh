#!/usr/bin/env bash
set -euo pipefail

COPA_V2_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COPA_V2_PACKAGE_ROOT="$(cd "${COPA_V2_SCRIPT_DIR}/.." && pwd)"
COPA_V2_REPOSITORY_ROOT="$(cd "${COPA_V2_PACKAGE_ROOT}/.." && pwd)"
COPA_V2_OUTPUT_DIR="${1:-${COPA_V2_PACKAGE_ROOT}/results/slate_multi_positive_v2}"
COPA_V2_WORKERS="${2:-4}"
COPA_V2_CONFIG_PATH="${3:?Usage: $0 [output_dir] [workers] /path/to/local/retrieval_extension.yaml}"

cd "${COPA_V2_REPOSITORY_ROOT}"

conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
  suite \
  --config "${COPA_V2_CONFIG_PATH}" \
  --output-dir "${COPA_V2_OUTPUT_DIR}" \
  --resume

conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
  evaluate-suite \
  --suite-dir "${COPA_V2_OUTPUT_DIR}" \
  --workers "${COPA_V2_WORKERS}" \
  --resume
