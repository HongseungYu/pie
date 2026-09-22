#!/usr/bin/env bash
# The RunPod pod that is the CUDA runner: start it before the GPU job, stop
# it after. Needs RUNPOD_API_KEY and RUNPOD_POD_ID.
#
#   runpod.sh start | stop | status
set -euo pipefail
api="https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID:?}"
call() { curl -fsS -X "$1" -H "Authorization: Bearer ${RUNPOD_API_KEY:?}" -H "Content-Type: application/json" "$api$2"; }
status() { call GET "" | jq -r '"\(.desiredStatus) gpu=\(.machine.gpuTypeId // .gpu // "-") uptime=\(.runtime.uptimeInSeconds // 0)s"'; }
case "${1:?start|stop|status}" in
  status) status ;;
  start)
    call POST /start >/dev/null
    # A stopped pod resumes only when its host has the GPU free; say so
    # rather than leaving the job to queue for a runner that never comes.
    for _ in $(seq 1 30); do
      s=$(status); echo "$s"
      case "$s" in RUNNING*) exit 0 ;; esac
      sleep 10
    done
    echo "pod ${RUNPOD_POD_ID} did not reach RUNNING" >&2; exit 1 ;;
  stop) call POST /stop >/dev/null; echo "pod ${RUNPOD_POD_ID} stopped" ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
