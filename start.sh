#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f "./target/release/cratevec-server" ]; then
    cargo build --release -p cratevec-server -p cratevec-cli
fi

SEED="${1:-PIXY - LEGACY}"
echo "Starting cratevec hybrid server on http://127.0.0.1:7420 (Seed: $SEED)..."
exec ./target/release/cratevec-server "$SEED"
