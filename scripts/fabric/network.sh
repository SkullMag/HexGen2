#!/usr/bin/env bash
# Keep the server under an explicit deadline; do not reuse another task's daemon.
set -euo pipefail
test "$#" -eq 3 || { echo 'usage: network.sh server|client PEER_DATA_IP FRESH_OUTPUT_DIR'; exit 2; }
task_mode=$1 task_peer=$2 task_output=$3
mkdir "$task_output"
task_finish() {
  task_status=$?
  ip -j -s link > "$task_output/links-after.json"
  date -u > "$task_output/end-utc.txt"
  echo "$task_status" > "$task_output/exit-code.txt"
}
trap task_finish EXIT
printf '%s\n' "$0 $*" > "$task_output/command.txt"
date -u > "$task_output/start-utc.txt"
ip -j -s link > "$task_output/links-before.json"
ip -j addr > "$task_output/addresses.json"
ip -j route > "$task_output/routes.json"
chronyc tracking > "$task_output/clock.txt" 2>&1 || true
ping -c 10 "$task_peer" > "$task_output/ping.txt" 2>&1
if [ "$task_mode" = server ]; then
  timeout 90 iperf3 -s --json > "$task_output/iperf-server.json" 2> "$task_output/iperf-server.err" || test "$?" -eq 124
elif [ "$task_mode" = client ]; then
  iperf3 -c "$task_peer" -t 10 -J > "$task_output/iperf-forward.json" 2> "$task_output/iperf-forward.err"
  iperf3 -c "$task_peer" -t 10 -R -J > "$task_output/iperf-reverse.json" 2> "$task_output/iperf-reverse.err"
else
  exit 2
fi
