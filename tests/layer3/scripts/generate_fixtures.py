#!/usr/bin/env python3
"""
generate_fixtures.py — Deterministic Set 1 Fixture Generator & SHA-256 Manifest Writer.

Generates controlled binary, histogram, parameter, and configuration fixtures under
`tests/layer3/fixtures/` and writes SHA-256 manifests to
`validation/layer3/fixtures_used/{fixture_hashes.json,generated_fixture_manifest.json}`.
"""

import argparse
import hashlib
import json
import os
import struct


def pack_u64(sx: int, sz: int, flags: int, epoch_id: int, seq_id: int) -> bytes:
    w = (
        (sx & 0xF)
        | ((sz & 0xF) << 4)
        | ((flags & 0xFF) << 8)
        | ((epoch_id & 0xFFFF) << 16)
        | ((seq_id & 0xFFFFFFFF) << 32)
    )
    return struct.pack("<Q", w)


def pack_yaml_header(magic: int, n_rounds: int, start_seq: int, nx=4, nz=4, dist=3, flags=1) -> bytes:
    return struct.pack("<IIIBBBB", magic, n_rounds, start_seq, nx, nz, dist, flags)


def pack_yaml_round(sx: int, sz: int, valid: int, meas_flip: int, reserved6: int, seq16: int) -> bytes:
    w = (
        (sx & 0xF)
        | ((sz & 0xF) << 4)
        | ((valid & 0x1) << 8)
        | ((meas_flip & 0x1) << 9)
        | ((reserved6 & 0x3F) << 10)
        | ((seq16 & 0xFFFF) << 16)
    )
    return struct.pack("<I", w)


def write_bytes(records: list, repo_root: str, rel_path: str, data: bytes, purpose: str):
    full = os.path.join(repo_root, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)
    records.append({
        "path": rel_path,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "purpose": purpose,
    })


def write_text(records: list, repo_root: str, rel_path: str, text: str, purpose: str):
    write_bytes(records, repo_root, rel_path, text.encode("utf-8"), purpose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/tmp/avid_layer3_work")
    args = ap.parse_args()
    r = args.repo_root
    records = []

    # 1. Packet fixtures: valid, invalid, boundary, expected
    write_bytes(records, r, "tests/layer3/fixtures/packets/valid/packet_001.bin",
                pack_u64(5, 11, 0x01, 3, 0),
                "L3-IP-001 canonical valid 64-bit packet (sx=5, sz=11, flags=VALID, epoch=3, seq=0)")

    seq_stream = b"".join(pack_u64((i * 3) & 0xF, (i * 5) & 0xF, 0x01, 1, i) for i in range(64))
    write_bytes(records, r, "tests/layer3/fixtures/packets/valid/burst_64_seq.bin",
                seq_stream,
                "64-packet valid monotonic DMA slot burst")

    illustrative_sx = [3, 3, 4, 3, 7]
    illustrative_sz = [1, 1, 2, 1, 5]
    illus_stream = b"".join(pack_u64(illustrative_sx[i], illustrative_sz[i], 0x01, 2, 100 + i) for i in range(5))
    write_bytes(records, r, "tests/layer3/fixtures/packets/valid/histogram_5pkt_stream.bin",
                illus_stream,
                "5-packet controlled stream (sx=[3,3,4,3,7], sz=[1,1,2,1,5])")

    yaml_frame = pack_yaml_header(0xAAEC0002, 2, 0) + pack_yaml_round(5, 11, 1, 0, 0, 0) + pack_yaml_round(3, 1, 1, 0, 0, 1)
    write_bytes(records, r, "tests/layer3/fixtures/packets/valid/yaml_v1_stream_2rounds.bin",
                yaml_frame,
                "syndrome_packet_v1.yaml 16B header + two 4B per-round words (24 bytes total)")

    # Invalid packets
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/empty_0B.bin", b"",
                "L3-IP-002 empty 0-byte packet buffer")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/truncated_3B.bin", b"\x5b\x01\x03",
                "L3-IP-009 3-byte truncated binary syndrome frame")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/short_7B.bin", b"\x05\x01\x00\x00\x00\x00\x00",
                "L3-IP-002 7-byte short packet buffer")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/oversize_9B.bin", pack_u64(5, 11, 0x01, 3, 0) + b"\xff",
                "L3-IP-002 9-byte packet buffer with trailing byte")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/yaml_hdr_truncated_15B.bin",
                pack_yaml_header(0xAAEC0002, 1, 0)[:15],
                "L3-IP-002 15-byte truncated YAML stream header")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/missing_valid_flag.bin",
                pack_u64(6, 9, 0x00, 1, 0),
                "L3-IP-004 packet with flags=0x00 (VALID=0)")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/parity_err_flag.bin",
                pack_u64(6, 9, 0x05, 1, 0),
                "L3-IP-008 packet with VALID|PARITY_ERR (0x05)")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/reserved_upper_flags.bin",
                pack_u64(6, 9, 0xF1, 1, 0),
                "L3-IP-004 packet with VALID|0xF0 reserved upper flag bits set")
    write_bytes(records, r, "tests/layer3/fixtures/packets/invalid/stim_parity_corrupted_64.bin",
                b"".join(pack_u64(0x6, 0x9, 0x04, 1, i) for i in range(64)),
                "64-packet stream where every packet has VALID=0 and PARITY_ERR=1 for daemon --stim-bin propagation test")

    # Boundary packets
    write_bytes(records, r, "tests/layer3/fixtures/packets/boundary/min_fields.bin",
                pack_u64(0, 0, 0x01, 0, 0),
                "Minimum valid fields (all zeros + VALID)")
    write_bytes(records, r, "tests/layer3/fixtures/packets/boundary/max_fields.bin",
                pack_u64(0xF, 0xF, 0x0B, 0xFFFF, 0xFFFFFFFF),
                "Maximum valid fields (sx=15, sz=15, VALID|OVERFLOW|EPOCH_MARK, epoch=65535, seq=UINT32_MAX)")
    rollover_pair = pack_u64(1, 2, 0x01, 1, 0xFFFFFFFF) + pack_u64(3, 4, 0x01, 1, 0)
    write_bytes(records, r, "tests/layer3/fixtures/packets/boundary/seq_rollover_pair.bin",
                rollover_pair,
                "UINT32_MAX -> 0 sequence rollover packet pair")

    # Expected decoded JSON
    write_text(records, r, "tests/layer3/fixtures/packets/expected/packet_001_expected.json",
               json.dumps({
                   "raw_u64_hex": "0x00000000000301b5",
                   "s_x": 5, "s_z": 11, "flags": 1, "epoch_id": 3, "seq_id": 0,
                   "valid": True
               }, indent=2) + "\n",
               "Expected decoded fields for packet_001.bin")

    # 2. Histogram fixtures
    write_text(records, r, "tests/layer3/fixtures/histograms/empty/zeros_16.txt",
               " ".join(["0"] * 16) + "\n", "L3-IP-013 empty 16-bin histogram")
    single_6 = ["64" if i == 6 else "0" for i in range(16)]
    write_text(records, r, "tests/layer3/fixtures/histograms/single_bin/bin6_64.txt",
               " ".join(single_6) + "\n", "L3-IP-014 single-state histogram at bin 6")
    write_text(records, r, "tests/layer3/fixtures/histograms/uniform/uniform_100.txt",
               " ".join(["100"] * 16) + "\n", "Uniform 16-bin histogram (100 per bin, total=1600)")
    sparse_3 = ["2" if i == 6 else ("1" if i == 0 else "0") for i in range(16)]
    write_text(records, r, "tests/layer3/fixtures/histograms/sparse/sparse_total3.txt",
               " ".join(sparse_3) + "\n", "L3-IP-029 sparse 3-round histogram")
    corr_hist = [800, 30, 35, 20, 30, 15, 25, 5, 20, 4, 8, 2, 4, 1, 1, 0]
    write_text(records, r, "tests/layer3/fixtures/histograms/correlated/corr_total1000.txt",
               " ".join(str(x) for x in corr_hist) + "\n", "L3-IP-015 correlated histogram summing to 1000")
    write_text(records, r, "tests/layer3/fixtures/histograms/correlated/corr_total10000.txt",
               " ".join(str(x * 10) for x in corr_hist) + "\n", "L3-IP-015 10x scaled histogram summing to 10000")
    write_text(records, r, "tests/layer3/fixtures/histograms/overflow/half_u32_16.txt",
               " ".join([str(0x7FFFFFFF)] * 16) + "\n", "L3-IP-017 sum > 2^32 histogram (16 * 2147483647)")

    # 3. Parameter fixtures
    valid_theta = [0.012] * 9 + [0.004] * 12
    write_text(records, r, "tests/layer3/fixtures/parameters/valid/default_21.txt",
               " ".join(f"{x:.6f}" for x in valid_theta) + "\n", "Nominal 21-parameter vector")
    write_text(records, r, "tests/layer3/fixtures/parameters/boundary/interior_bounds_21.txt",
               " ".join([f"{1e-5:.8f}"] * 9 + [f"{0.45:.6f}"] * 12) + "\n", "Interior clamp boundary vector")
    write_text(records, r, "tests/layer3/fixtures/parameters/invalid/out_of_range_21.txt",
               " ".join(["-0.001"] + ["0.51"] + ["0.012"] * 19) + "\n", "Out-of-range probabilities (<0 and >0.5)")
    write_text(records, r, "tests/layer3/fixtures/parameters/priors/default_prior_21.txt",
               " ".join(f"{x:.6f}" for x in valid_theta) + "\n", "Default 21-parameter prior vector")

    # 4. Configuration fixtures (valid, invalid, boundary YAML)
    valid_cfg = """experiment:
  name: layer3_set1_valid
  seed: 2026
  timeout_seconds: 30
  rounds_per_scenario: 1024
  window_rounds: 256
software:
  emulator: native
  architecture: x86_64
  boot_mode: direct
  daemon: avid_daemon
  hal_backend: physical
  controller_mode: heuristic
  osd_mode: 1
rtl:
  simulator: fallback
  top_module: aa_qec_top
  clock_period_ns: 4
  reset_cycles: 10
  bp_iterations: 3
  waveform: false
bridge:
  transport: unix_socket
  protocol_version: 1
  clock_mode: cycle_exact
  transaction_timeout_ms: 2000
scenarios:
  - stationary_noise
validation:
  assertions: true
  fail_on_rtl_error: true
  fail_on_timeout: true
artifacts:
  output_dir: validation/layer3
  save_logs: true
  save_waveforms: false
  save_transactions: true
  save_metrics: true
"""
    write_text(records, r, "tests/layer3/fixtures/configs/valid/valid_experiment.yaml",
               valid_cfg, "Valid configuration matching configs/schema.yaml")

    invalid_unknown_cfg = valid_cfg + "\nunknown_top_section:\n  rogue_flag: true\n"
    write_text(records, r, "tests/layer3/fixtures/configs/invalid/unknown_key_experiment.yaml",
               invalid_unknown_cfg, "Invalid configuration containing unknown top-level key")

    invalid_missing_cfg = """experiment:
  name: missing_fields_test
software:
  daemon: avid_daemon
"""
    write_text(records, r, "tests/layer3/fixtures/configs/invalid/missing_required.yaml",
               invalid_missing_cfg, "Invalid configuration missing required sections and keys")

    boundary_cfg = valid_cfg.replace("window_rounds: 256", "window_rounds: 64").replace("bp_iterations: 3", "bp_iterations: 5")
    write_text(records, r, "tests/layer3/fixtures/configs/boundary/min_window_max_k.yaml",
               boundary_cfg, "Boundary configuration (window_rounds=64, bp_iterations=5)")

    hashes_map = {rec["path"]: rec["sha256"] for rec in records}
    out_hashes = os.path.join(r, "validation/layer3/fixtures_used/fixture_hashes.json")
    out_manifest = os.path.join(r, "validation/layer3/fixtures_used/generated_fixture_manifest.json")
    with open(out_hashes, "w", encoding="utf-8") as wf:
        json.dump(hashes_map, wf, indent=2, sort_keys=True)
        wf.write("\n")
    with open(out_manifest, "w", encoding="utf-8") as wf:
        json.dump({
            "schema_version": "1.0.0",
            "generator": "tests/layer3/scripts/generate_fixtures.py",
            "fixture_count": len(records),
            "fixtures": records,
        }, wf, indent=2)
        wf.write("\n")

    print(f"Generated {len(records)} deterministic fixtures and SHA-256 manifests.")


if __name__ == "__main__":
    main()
