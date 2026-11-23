#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
runtime="${GO2_RUNTIME_DIR:-$HOME/.cache/go2-piper-supermimic-v2}"
export PYTHONPATH="$PWD/vendor/supermimic-quadruped-0.2.0/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONUNBUFFERED=1
if [[ "${1:-}" == joint ]]; then
    shift
    exec "$runtime/venv/bin/python" -m experiments.walk_stop_reach.joint "$@"
fi
exec "$runtime/venv/bin/python" -m experiments.walk_stop_reach.experiment "$@"
