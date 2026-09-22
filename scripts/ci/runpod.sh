#!/usr/bin/env bash
# The RunPod pod that is the CUDA runner: start it before the GPU job, stop
# it after. Needs RUNPOD_API_KEY and RUNPOD_POD_ID.
#
#   runpod.sh start | stop | status
set -euo pipefail
api="https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID:?}"
call() { curl -fsS -X "$1" -H "Authorization: Bearer ${RUNPOD_API_KEY:?}" -H "Content-Type: application/json" "$api$2"; }
case "${1:?start|stop|status}" in
  status) call GET "" | jq -r '"\(.desiredStatus) \(.machine.gpuTypeId // "-") \(.name)"' ;;
  start)  call POST /start >/dev/null; echo "pod ${RUNPOD_POD_ID} starting" ;;
  stop)   call POST /stop >/dev/null; echo "pod ${RUNPOD_POD_ID} stopped" ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
