#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
mkdir -p "${REPO_ROOT}/validation/layer3/manifests"

CC_VER="$(cc --version | head -n 1)"
CXX_VER="$(c++ --version | head -n 1)"
CMAKE_VER="$(cmake --version | head -n 1)"
CTEST_VER="$(ctest --version | head -n 1)"
PY_VER="$(python3 --version)"
UNAME_STR="$(uname -srmo)"

python3 - <<PYEOF
import json, datetime

env_doc = {
    "schema_version": "1.0.0",
    "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "os_kernel": "${UNAME_STR}",
    "c_compiler": "${CC_VER}",
    "cxx_compiler": "${CXX_VER}",
    "cmake": "${CMAKE_VER}",
    "ctest": "${CTEST_VER}",
    "python": "${PY_VER}",
    "c_standard": "C11 (-std=c11, extensions OFF)",
    "cxx_standard": "C++17 (-std=c++17, extensions OFF)",
}
with open("${REPO_ROOT}/validation/layer3/manifests/environment_manifest.json", "w", encoding="utf-8") as f:
    json.dump(env_doc, f, indent=2)
    f.write("\n")
PYEOF

echo "[verify_environment] OK: ${CXX_VER} | ${CMAKE_VER} | ${PY_VER}"
