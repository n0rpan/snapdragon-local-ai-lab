#!/usr/bin/env bash
# Three fixed-workload session replays of ONE recording for a promotion batch (operator runs this; it needs
# the hardware). Each replay uses a FRESH bare oxe-server that owns the canonical inference lock for its whole
# life, inside a systemd user scope (MemoryMax=28G, MemorySwapMax=0), a private empty prefix dir, no spec decoding,
# OXE_ATTN_TILED_MIN_KV=0, decode cpus 0-7, prefill cpus 0-11, ubatch 256.
#   1 base-ornith-16k : engine 649ac806, Ornith,        ctx 16384, ck 4
#   2 base-qwen-16k   : engine 649ac806, Qwen3.6-35B,   ctx 16384, ck 4
#   3 cand-ornith-32k : engine b9ad2eb05c33, Ornith,    ctx 32768, ck 8, --session-ram-bytes 2147483648
# Raw replay JSON (contains prompts) goes to a private dir OUTSIDE git; a numeric-only extract goes to PUBLIC_OUT.
# Usage: tools/run_batch_replays.sh [recording.jsonl]   (env overrides below). Nothing here is run by the author.
set -uo pipefail
H="$HOME"
REC="${RECORDING:-$H/.local/state/local-ai/recordings/ornith-m2-queue-batch1-20261110/capture/recording.jsonl}"
[ -n "${1:-}" ] && [ "$1" != "--inner" ] && REC="$1"
REPLAY="${REPLAY_TOOL:-$(cd "$(dirname "$0")" && pwd)/session_replay.py}"
REL="$H/.local/share/ai-stack/releases"
BASE_BIN="${BASE_BIN:-$REL/oxe-649ac80670e5/oxe-server}"
CAND_BIN="${CAND_BIN:-$REL/oxe-p1-b9ad2eb05c33/oxe-server}"
MODELS="$H/.local/share/ai-stack/models/llm"
ORNITH="${ORNITH_MODEL:-$MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf}"
QWEN="${QWEN_MODEL:-$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf}"
LOCK="$H/.local/state/local-ai/inference.lock"
PORT="${PORT:-8097}"
STAMP="$(date +%Y%m%d-%H%M%S)"
PRIVATE="${PRIVATE_OUT:-$H/.local/state/local-ai/replays/batch-$STAMP}"   # raw: never git
PUBLIC_OUT="${PUBLIC_OUT:-$PRIVATE/public-extract.json}"                   # numeric only; copy to git by hand
mkdir -p -m 700 "$PRIVATE"
[ -f "$REC" ] || { echo "recording not found: $REC" >&2; exit 2; }
[ -f "$REPLAY" ] || { echo "replay tool not found: $REPLAY" >&2; exit 2; }
for f in "$BASE_BIN" "$CAND_BIN" "$ORNITH" "$QWEN"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }; done

# label | binary | model | alias | ctx | extra server args
RUNS=(
  "base-ornith-16k|$BASE_BIN|$ORNITH|ornith-35b-a3b-local|16384|--ctx-checkpoints 4"
  "base-qwen-16k|$BASE_BIN|$QWEN|qwen3.6-35b-a3b-local|16384|--ctx-checkpoints 4"
  "cand-ornith-32k|$CAND_BIN|$ORNITH|ornith-35b-a3b-local|32768|--ctx-checkpoints 8 --session-ram-bytes 2147483648"
)

run_one() {  # runs inside the scope while holding the lock; $1..: label bin model alias ctx extra...
  local label="$1" bin="$2" model="$3" alias="$4" ctx="$5"; shift 5
  local dir="$PRIVATE/$label"; mkdir -p -m 700 "$dir/prefix"
  export OXE_ATTN_TILED_MIN_KV=0
  "$bin" --engine oxe --model "$model" --alias "$alias" --ctx-size "$ctx" --host 127.0.0.1 --port "$PORT" \
    --oxe-cpus 0-7 --oxe-prefill-cpus 0-11 --ubatch-size 256 --reasoning off --spec-type none \
    --prefix-cache "$dir/prefix" "$@" >"$dir/server.log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || break
    curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 1
  done
  if [ "$up" != 1 ]; then echo "server did not come up ($label)" >&2; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 3; fi
  local sha; sha="$(sha256sum "$bin" | cut -d' ' -f1)"
  python3 "$REPLAY" replay --server "http://127.0.0.1:$PORT" --out "$dir/replay.json" --label "$label" "$REC" \
    >"$dir/replay.log" 2>&1
  local rc=$?
  local rss hwm swap
  rss="$(awk '/^VmRSS:/{print $2*1024}' /proc/$pid/status 2>/dev/null)"
  hwm="$(awk '/^VmHWM:/{print $2*1024}' /proc/$pid/status 2>/dev/null)"
  swap="$(awk '/^VmSwap:/{print $2*1024}' /proc/$pid/status 2>/dev/null)"
  printf '{"label":"%s","engine_sha256":"%s","ctx":%s,"vm_rss":%s,"vm_hwm":%s,"vm_swap":%s,"replay_rc":%s}\n' \
    "$label" "$sha" "$ctx" "${rss:-null}" "${hwm:-null}" "${swap:-null}" "$rc" >"$dir/engine.json"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  return "$rc"
}

if [ "${1:-}" = "--inner" ]; then shift; run_one "$@"; exit $?; fi

status=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r label bin model alias ctx extra <<<"$spec"
  echo "== $label"
  # shellcheck disable=SC2086
  flock -n -E 75 "$LOCK" systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=0 -- \
    env REC="$REC" REPLAY_TOOL="$REPLAY" PRIVATE_OUT="$PRIVATE" PORT="$PORT" BASE_BIN="$BASE_BIN" CAND_BIN="$CAND_BIN" \
        ORNITH_MODEL="$ORNITH" QWEN_MODEL="$QWEN" \
    "$0" --inner "$label" "$bin" "$model" "$alias" "$ctx" $extra
  rc=$?
  [ "$rc" = 75 ] && { echo "inference lock busy; stopping" >&2; exit 75; }
  [ "$rc" != 0 ] && status=$rc
done

# Public numeric extract: only numbers/ids from the raw results, no text. Raw stays in $PRIVATE.
python3 - "$PRIVATE" "$PUBLIC_OUT" <<'PY'
import json, sys, pathlib
priv, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def numeric(o):
    if isinstance(o, bool) or o is None: return o
    if isinstance(o, (int, float)): return o
    if isinstance(o, dict): return {k: numeric(v) for k, v in o.items() if numeric(v) not in ({}, [], '')}
    if isinstance(o, list): return [v for v in map(numeric, o) if v not in ({}, [], '')]
    return ''  # drop every string (prompts, replies, paths)
rows = []
for d in sorted(p for p in priv.iterdir() if p.is_dir()):
    row = {'label': d.name}
    for name in ('engine.json', 'replay.json'):
        f = d / name
        if f.exists():
            row[name.split('.')[0]] = numeric(json.loads(f.read_text())) if name == 'replay.json' else json.loads(f.read_text())
    rows.append(row)
out.write_text(json.dumps(rows, indent=1) + '\n')
print('public extract:', out)
PY
exit $status
