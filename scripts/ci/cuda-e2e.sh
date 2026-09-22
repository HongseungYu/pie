#!/usr/bin/env bash
# pie on a real GPU, end to end: a debug `pie` with the CUDA engine serves
# Qwen3.5-0.8B; the compat API suite runs against it; an inferlet is
# installed, replaced and removed while it runs; `pie run` resolves a bare
# name. PIE_HOME holds the model between runs (the runner's volume).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PIE_HOME="${PIE_HOME:-$HOME/.pie}"
port=${PIE_E2E_PORT:-18517}
pie=target/debug/pie

cargo build -p pie --features cuda --bin pie
(cd examples && cargo build -q --release --target wasm32-wasip2 -p text-completion -p naive-baseline)

if ! "$pie" model list 2>/dev/null | grep -q 'Qwen3.5-0.8B'; then
  "$pie" model import Qwen/Qwen3.5-0.8B
fi
mkdir -p "$PIE_HOME"
cat > "$PIE_HOME/config.toml" <<TOML
[server]
host = "127.0.0.1"
port = $port
telemetry = false
[model]
name = "default"
model = "Qwen/Qwen3.5-0.8B"
[engine]
type = "cuda_native"
device = ["cuda:0"]
activation_dtype = "bfloat16"
gpu_mem_utilization = 0.85
[sandbox]
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]
TOML
"$pie" doctor || true

"$pie" inferlet remove text-completion@0.3.0 >/dev/null 2>&1 || true
log="${RUNNER_TEMP:-/tmp}/pie-serve.log"
"$pie" serve > "$log" 2>&1 &
serve=$!
trap 'kill $serve 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do grep -q "Server ready\|✗" "$log" && break; sleep 2; done
grep -q "Server ready" "$log" || { echo "the server did not come up"; tail -30 "$log"; exit 1; }

echo "== compat API suite"
uv run --with openai --with anthropic --with google-genai python tests/builtins/test_compat.py --base-url "http://127.0.0.1:$port"

echo "== install, replace and remove while serving"
submit() { uv run --project python/client python - "$@" <<'PY'
import asyncio, sys
from pie_client import PieClient
async def main():
    async with PieClient(f"ws://127.0.0.1:{sys.argv[2]}") as c:
        await c.authenticate("ci", None)
        try:
            p = await c.launch_process(sys.argv[1], {"prompt": "The capital of France is", "max_tokens": 4})
            print("ok", str(await p.result())[:60])
        except Exception as e:
            print("error", str(e).splitlines()[0][:80])
asyncio.run(main())
PY
}
tc=examples/target/wasm32-wasip2/release/text_completion.wasm
nb=examples/target/wasm32-wasip2/release/naive_baseline.wasm
man=examples/text-completion/Pie.toml
submit text-completion "$port" | grep -q '^error' || { echo "expected: not installed"; exit 1; }
"$pie" inferlet install "$tc" -m "$man"
submit text-completion "$port" | grep -q "^ok.*Paris" || { echo "expected: Paris"; exit 1; }
"$pie" inferlet install "$nb" -m "$man" --force
submit text-completion "$port" | grep -q "^ok.*sampler" || { echo "expected: the replacement's output"; exit 1; }
"$pie" inferlet remove text-completion@0.3.0
submit text-completion "$port" | grep -q '^error' || { echo "expected: removed"; exit 1; }
if grep -q 'panicked at' "$log"; then echo "the server panicked"; grep -A3 'panicked at' "$log"; exit 1; fi
kill $serve; wait $serve 2>/dev/null || true

echo "== pie run by bare name"
"$pie" run text-completion | grep -q Paris
echo "cuda e2e: ok"
