#!/bin/sh
# Run the lab unit suite with a throwaway HOME and XDG roots (PATH kept), so no default path can
# reach the real ~/.config/local-ai, ~/.local/state/local-ai or the Omarchy panel. Extra
# arguments go to unittest (default: discover -s tests).
set -eu
cd "$(dirname "$0")/.."
root=$(mktemp -d "${TMPDIR:-/tmp}/local-ai-lab-tests-XXXXXX")
trap 'rm -rf "$root"' EXIT
mkdir -m 700 "$root/home" "$root/config" "$root/state" "$root/data" "$root/cache" "$root/runtime"
if [ "$#" -eq 0 ]; then set -- discover -s tests; fi
env -u LOCAL_LLM_CONFIG -u LOCAL_AI_LEASE HOME="$root/home" XDG_CONFIG_HOME="$root/config" \
    XDG_STATE_HOME="$root/state" XDG_DATA_HOME="$root/data" XDG_CACHE_HOME="$root/cache" \
    XDG_RUNTIME_DIR="$root/runtime" PYTHONPATH=src nice -n 19 python3 -m unittest "$@"
