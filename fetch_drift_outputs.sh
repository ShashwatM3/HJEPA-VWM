#!/usr/bin/env bash
# Fetch drift-probe outputs (PNGs + JSON, not the ~1 GB feature cache) from a
# RunPod pod to this machine, over RunPod's PROXY SSH (<pod-id>@ssh.runpod.io).
#
# Why this dance: the proxy REQUIRES a PTY, IGNORES the ssh exec command
# argument (commands must be piped via stdin), and the PTY pollutes output with
# banners, echoed input, ANSI escapes, and CRLFs — so scp/sftp/cat pipes all
# fail. Bytes therefore travel as marker-delimited base64: everything outside
# the markers is discarded, and base64 survives CRLF stripping untouched.
# (If your Connect tab offers a DIRECT TCP command — ssh root@<ip> -p <port> —
# plain `scp -P <port>` works over that instead and you don't need this script.)
#
# Usage:
#   ./fetch_drift_outputs.sh <ssh-target> [identity-file] [dest-dir]
# Example:
#   ./fetch_drift_outputs.sh <pod-id>@ssh.runpod.io ~/.ssh/id_ed25519 ~/Desktop
#
# Extracts to <dest-dir>/drift_probe/.
set -euo pipefail

TARGET=${1:?usage: fetch_drift_outputs.sh <ssh-target> [identity-file] [dest-dir]}
KEY=${2:-$HOME/.ssh/id_ed25519}
DEST=${3:-$HOME/Desktop}
REMOTE_LOGS=/workspace/hierarchal-jepa-flow-world-model/logs

mkdir -p "$DEST"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# stty -echo: stop the PTY echoing our command back into the stream.
# Quoted marker halves (__B64_"START"__): the echoed/wrapped command line can
# never contain the assembled marker string, so marker matching cannot false-fire.
# Remote stderr -> /dev/null: on a PTY, stderr would interleave INTO stdout
# mid-stream and corrupt the base64 payload.
printf 'stty -echo; echo __B64_"START"__; tar -C %s -czf - --exclude="*.pt" --exclude="*.tmp" drift_probe 2>/dev/null | base64; echo __B64_"END"__; exit\n' "$REMOTE_LOGS" \
  | ssh -tt -o ConnectTimeout=15 "$TARGET" -i "$KEY" 2>/dev/null \
  | tr -d '\r' \
  | awk 'index($0,"__B64_END__"){f=0} f{print} index($0,"__B64_START__"){f=1}' \
  | base64 -d > "$TMP"

if [ ! -s "$TMP" ]; then
  echo "ERROR: received no payload. Check the ssh target/key, and that" >&2
  echo "  $REMOTE_LOGS/drift_probe exists on the pod." >&2
  exit 1
fi

tar -xzf "$TMP" -C "$DEST"
echo "Drift-probe outputs extracted to: $DEST/drift_probe"
ls -l "$DEST/drift_probe"
