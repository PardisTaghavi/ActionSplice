#!/usr/bin/env bash
set -euo pipefail

required=(
  ACTIONSPLICE_MINWM_CST_R_REPO
  ACTIONSPLICE_MINWM_CST_T_REPO
  ACTIONSPLICE_HY_CST_R_REPO
  ACTIONSPLICE_HY_CST_T_REPO
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" || "${!name}" == *"<"* ]]; then
    echo "CST weights are not public yet; set ${name} after release." >&2
    exit 2
  fi
done

weight_root="${1:-checkpoints}"
mkdir -p "${weight_root}"

hf download "${ACTIONSPLICE_MINWM_CST_R_REPO}" --local-dir "${weight_root}/minwm-cst-r"
hf download "${ACTIONSPLICE_MINWM_CST_T_REPO}" --local-dir "${weight_root}/minwm-cst-t"
hf download "${ACTIONSPLICE_HY_CST_R_REPO}" --local-dir "${weight_root}/hyworld15-cst-r"
hf download "${ACTIONSPLICE_HY_CST_T_REPO}" --local-dir "${weight_root}/hyworld15-cst-t"
