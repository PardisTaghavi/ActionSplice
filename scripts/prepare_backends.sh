#!/usr/bin/env bash
set -euo pipefail

BACKEND_ROOT="${1:-third_party}"
mkdir -p "${BACKEND_ROOT}"

if [[ ! -d "${BACKEND_ROOT}/minWM/.git" ]]; then
  git clone https://github.com/shengshu-ai/minWM.git "${BACKEND_ROOT}/minWM"
fi
git -C "${BACKEND_ROOT}/minWM" fetch origin main
git -C "${BACKEND_ROOT}/minWM" checkout df522a2

if [[ ! -d "${BACKEND_ROOT}/HY-WorldPlay/.git" ]]; then
  git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git \
    "${BACKEND_ROOT}/HY-WorldPlay"
fi
git -C "${BACKEND_ROOT}/HY-WorldPlay" fetch origin main
git -C "${BACKEND_ROOT}/HY-WorldPlay" checkout \
  1588e1336e842b03b0a7860c654ebd7c46bb065e

echo "Prepared pinned backends under ${BACKEND_ROOT}"
