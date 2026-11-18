#!/usr/bin/env bash
# Generate from sandboxd's authoritative specification in a platform checkout.
set -euo pipefail
platform_root=${1:?Usage: generate-command-session.sh /path/to/platform}
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
proto_root=$(mktemp -d)
trap 'rm -rf -- "$proto_root"' EXIT
mkdir -p "$proto_root/command_session"
cp "$platform_root/sandbox/vm-sandboxes/packages/sandboxd/spec/command_session.proto" "$proto_root/command_session/command_session.proto"
uv run --no-project --python 3.13 --with grpcio-tools==1.73.1 python -m grpc_tools.protoc \
  -I "$proto_root" --python_out="$package_root/src/prime_sandboxes/_proto" \
  command_session/command_session.proto
