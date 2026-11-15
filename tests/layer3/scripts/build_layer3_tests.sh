#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
STD_BUILD="${REPO_ROOT}/build/layer3_standard"
SAN_BUILD="${REPO_ROOT}/build/layer3_sanitizers"
LOG_DIR="${REPO_ROOT}/validation/layer3/logs/set1"
mkdir -p "${LOG_DIR}" "${REPO_ROOT}/validation/layer3/manifests"

echo "[build_layer3_tests] 1/2 Building Standard C11/C++17 suite -> ${STD_BUILD}"
cmake -S "${REPO_ROOT}/tests/layer3" -B "${STD_BUILD}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DAVID_ENABLE_ASAN=OFF \
    -DAVID_ENABLE_UBSAN=OFF > "${LOG_DIR}/cmake_configure_standard.log" 2>&1
cmake --build "${STD_BUILD}" -j4 > "${LOG_DIR}/cmake_build_standard.log" 2>&1

echo "[build_layer3_tests] 2/2 Building ASan+UBSan C11/C++17 suite -> ${SAN_BUILD}"
cmake -S "${REPO_ROOT}/tests/layer3" -B "${SAN_BUILD}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DAVID_ENABLE_ASAN=ON \
    -DAVID_ENABLE_UBSAN=ON > "${LOG_DIR}/cmake_configure_sanitizers.log" 2>&1
cmake --build "${SAN_BUILD}" -j4 > "${LOG_DIR}/cmake_build_sanitizers.log" 2>&1

python3 - <<PYEOF
import json, os, datetime

repo_root = "${REPO_ROOT}"
comp_db_path = os.path.join("${STD_BUILD}", "compile_commands.json")
with open(comp_db_path, "r", encoding="utf-8") as f:
    cmds = json.load(f)

compiled_files = sorted({
    os.path.relpath(entry["file"], repo_root)
    for entry in cmds
})
prod_files = [p for p in compiled_files if p.startswith(("software/", "deployment/"))]
c_abi_files = [p for p in compiled_files if "/c_abi/" in p]
test_cpp_files = [p for p in compiled_files if p.startswith("tests/layer3/") and "/c_abi/" not in p]

doc = {
    "schema_version": "1.0.0",
    "built_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "standard_build_status": "PASS",
    "asan_ubsan_build_status": "PASS",
    "total_unique_translation_units_compiled": len(compiled_files),
    "production_sources_compiled": prod_files,
    "c_abi_sources_compiled": c_abi_files,
    "cpp_harness_and_set1_sources_compiled": test_cpp_files,
    "all_compiled_files": compiled_files,
}
with open(os.path.join(repo_root, "validation/layer3/manifests/build_manifest.json"), "w", encoding="utf-8") as wf:
    json.dump(doc, wf, indent=2)
    wf.write("\n")
print(f"[build_layer3_tests] Recorded {len(compiled_files)} compiled translation units ({len(prod_files)} production, {len(c_abi_files)} C_ABI, {len(test_cpp_files)} C++ Set 1/common).")
PYEOF
