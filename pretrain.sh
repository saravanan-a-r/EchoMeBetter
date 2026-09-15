#!/usr/bin/env bash
# Full pretrain run: 90,000 steps per training_config.yml's pretrain stage,
# with its TensorBoard dashboard served on this machine at localhost:6006.
#
# View the dashboard from the Mac through an SSH tunnel -- either add
# `-L 6006:localhost:6006` to the usual login:
#   ssh -i <path-to-your-key> -L 8888:localhost:8888 -L 6006:localhost:6006 <user>@<gpu-box-host>
# or open a tunnel-only session in a separate Mac terminal:
#   ssh -i <path-to-your-key> -N -L 6006:localhost:6006 <user>@<gpu-box-host>
# then browse to http://localhost:6006 on the Mac.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TB_DIR=runs/pretrain/tensorboard
TB_PORT=6006
# The viewer has its own venv so its dependencies (it still needs setuptools'
# pkg_resources) can never disturb the training environment. One-time setup:
#   /root/venv/bin/python -m venv /root/tensorboard-venv
#   /root/tensorboard-venv/bin/pip install "tensorboard==2.20.0" "setuptools<81"
TB_BIN=/root/tensorboard-venv/bin/tensorboard

# TensorBoard only reads the event files training writes, so it can be stopped
# or restarted at any time without touching training. It runs in its own
# session (a Ctrl-C to training leaves it up), at the lowest CPU priority,
# bound to localhost so only the SSH tunnel reaches it; a re-run of this script
# after a crash finds the port already served and leaves it be.
# samples_per_plugin keeps every point: TensorBoard otherwise keeps a random
# 1,000 per chart, which over 90,000 steps can sample a one-step loss spike
# away. (0 would mean "keep none" in this version, not "keep all".)
# Start with the Custom Scalars tab; the Text tab's `guide` explains every
# metric and links to the Time Series overview.
mkdir -p "$TB_DIR" pretrain_corpus/logs
if ss -Hltn "sport = :$TB_PORT" | grep -q .; then
    echo "dashboard: port $TB_PORT already served, reusing it"
elif [ -x "$TB_BIN" ]; then
    setsid nohup nice -n 19 "$TB_BIN" --logdir "$TB_DIR" \
        --host 127.0.0.1 --port "$TB_PORT" \
        --samples_per_plugin scalars=100000,text=1000 --reload_interval 30 \
        --window_title "EchoMeBetter pretrain" \
        >> pretrain_corpus/logs/tensorboard.log 2>&1 < /dev/null &
else
    echo "dashboard: $TB_BIN not found (see the one-time setup above); training continues, event files are still written"
fi
echo "dashboard: http://localhost:$TB_PORT  (tunnel from the Mac: ssh -i <path-to-your-key> -N -L $TB_PORT:localhost:$TB_PORT <user>@<gpu-box-host>)"

/root/venv/bin/python pretrain.py --stage pretrain --device cuda \
    --corpus pretrain_corpus/output/pretrain_output \
    --quick-eval-corpus pretrain_corpus/output/evals/quick_eval.jsonl \
    --master-eval-corpus pretrain_corpus/output/evals/master_eval.jsonl \
    --log-file pretrain_corpus/logs/pretrain.log \
    --tensorboard-dir "$TB_DIR"
