#!/bin/bash
# XPU tests. Inside the container:   /opt/xpu-tests/run.sh
#   from the host:                    docker exec ai-toolkit-xpu /opt/xpu-tests/run.sh
# Optional end-to-end LoRA check on a real job config (slow, downloads the model):
#   docker exec ai-toolkit-xpu /opt/xpu-tests/run.sh /app/ai-toolkit/output/<job>/.job_config.json
set -u
cd "$(dirname "$0")"
fail=0
log=$(mktemp)
run() {
    echo "=== $1"
    shift
    "$@" > "$log" 2>&1
    local status=$?
    grep -v -E "UserWarning|warnings.warn|^\s*$" "$log"
    [ $status -eq 0 ] || { echo "(exit code $status)"; fail=1; }
}
run "attention (exact kernel, hybrid routing)" env AITK_XPU=1 python3 test_attention.py
run "non-finite gradient guard" env AITK_XPU=1 python3 test_nan_guard.py
run "layer-offload prefetch" env AITK_XPU=1 python3 test_prefetch.py
if [ $# -ge 1 ]; then
    run "end-to-end LoRA training: $1" python3 validate_model.py "$1" "${2:-30}"
fi
rm -f "$log"
echo
[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail
