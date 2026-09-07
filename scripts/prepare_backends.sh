#!/usr/bin/env bash
set -euo pipefail

BACKEND_ROOT="${1:-third_party}"
MINWM_REVISION="df522a26cd4409d3e3e8f269cc98eac069b5df47"
HYWORLD_REVISION="1588e1336e842b03b0a7860c654ebd7c46bb065e"
mkdir -p "${BACKEND_ROOT}"

if [[ ! -d "${BACKEND_ROOT}/minWM/.git" ]]; then
  git clone https://github.com/shengshu-ai/minWM.git "${BACKEND_ROOT}/minWM"
fi
git -C "${BACKEND_ROOT}/minWM" fetch origin main
git -C "${BACKEND_ROOT}/minWM" checkout --detach "${MINWM_REVISION}"

if [[ ! -d "${BACKEND_ROOT}/HY-WorldPlay/.git" ]]; then
  git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git \
    "${BACKEND_ROOT}/HY-WorldPlay"
fi
git -C "${BACKEND_ROOT}/HY-WorldPlay" fetch origin main
git -C "${BACKEND_ROOT}/HY-WorldPlay" checkout --detach "${HYWORLD_REVISION}"

echo "Prepared pinned backend source checkouts under ${BACKEND_ROOT}"
