#!/usr/bin/env bash
# Serve the SFT runs' TensorBoard on its own port, away from pretraining's
# (6006, runs/pretrain/tensorboard). Pinned to the spare cores 22-27 at
# nice 19 so it never competes with pretraining or a CPU SFT run.
#
#   SFT_adapter_rephrase/tensorboard.sh            # port 6007
#   PORT=6010 SFT_adapter_rephrase/tensorboard.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-6007}"
LOGDIR="${HERE}/logs"
TENSORBOARD="${TENSORBOARD:-/root/tensorboard-venv/bin/tensorboard}"
if [[ ! -x "${TENSORBOARD}" ]]; then
  TENSORBOARD="$(command -v tensorboard)"
fi

mkdir -p "${LOGDIR}"
exec nice -n 19 taskset -c 22-27 "${TENSORBOARD}" \
  --logdir "${LOGDIR}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --samples_per_plugin scalars=100000,text=1000,histograms=200 \
  --reload_interval 30 \
  --window_title "EchoMeBetter SFT rephrase adapter"
