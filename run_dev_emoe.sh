#!/usr/bin/env bash
set -euo pipefail

DEV_EMOE_DIR="${DEV_EMOE_DIR:-/liuzongfang/Projects/dev-emoe}"
HF_ENV_SCRIPT="${HF_ENV_SCRIPT:-/liuzongfang/hf_env.sh}"
VENV_ACTIVATE_REL="${VENV_ACTIVATE_REL:-.venv/bin/activate}"

usage() {
  cat <<'EOF'
Usage: scripts/run_dev_emoe.sh <command> [args...]

Runs the given command after:
  - cd to /liuzongfang/Projects/dev-emoe
  - bash /liuzongfang/hf_env.sh
  - source .venv/bin/activate

Overrides:
  DEV_EMOE_DIR, HF_ENV_SCRIPT, VENV_ACTIVATE_REL
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

if [[ ! -d "$DEV_EMOE_DIR" ]]; then
  echo "Error: DEV_EMOE_DIR not found: $DEV_EMOE_DIR" >&2
  exit 1
fi

cd "$DEV_EMOE_DIR"

if [[ ! -f "$HF_ENV_SCRIPT" ]]; then
  echo "Error: HF_ENV_SCRIPT not found: $HF_ENV_SCRIPT" >&2
  exit 1
fi

bash "$HF_ENV_SCRIPT"

VENV_ACTIVATE="$DEV_EMOE_DIR/$VENV_ACTIVATE_REL"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Error: venv activate script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

exec "$@"
