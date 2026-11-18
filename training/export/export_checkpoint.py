"""Export trained R2D2 V1/V2 networks to C++ binary (.bin), metadata (.json), and ONNX (.onnx)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple
import numpy as np

from training.models.r2d2_network import R2D2Network
from training.observations.observation_schema import FEATURE_NAMES_82, OBS_V1_DIM, OBS_V2_DIM


def tensor_order_for_dims(
    input_dim: int = OBS_V2_DIM,
    hidden_dim: int = 64,
    num_actions: int = 6,
    num_qubits: int = 9,
) -> List[Tuple[str, Tuple[int, ...]]]:
    """Return canonical binary tensor serialization layout for `input_dim`."""
    d = int(input_dim)
    h = int(hidden_dim)
    a = int(num_actions)
    q = int(num_qubits)
    return [
        ("feature_scale", (d,)),
        ("feature_offset", (d,)),
        ("W_ih", (4 * h, d)),
        ("W_hh", (4 * h, h)),
        ("b_ih", (4 * h,)),
        ("W_val", (1, h)),
        ("b_val", (1,)),
        ("W_adv", (a, h)),
        ("b_adv", (a,)),
        ("W_gain", (q, h)),
        ("b_gain", (q,)),
    ]


# Default V2 tensor order (82-input R2D2 V2)
TENSOR_ORDER: List[Tuple[str, Tuple[int, ...]]] = tensor_order_for_dims(OBS_V2_DIM)
TENSOR_ORDER_V1: List[Tuple[str, Tuple[int, ...]]] = tensor_order_for_dims(OBS_V1_DIM)


def infer_schema_version_for_input_dim(input_dim: int) -> int:
    if int(input_dim) == OBS_V2_DIM:
        return 2
    if int(input_dim) == OBS_V1_DIM:
        return 1
    raise ValueError(f"Unsupported R2D2 checkpoint input_dim={input_dim}; expected 82 (V2) or 8 (V1)")


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    v = int(value)
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(field_number: int, payload: bytes) -> bytes:
    return _encode_tag(field_number, 2) + _encode_varint(len(payload)) + payload


def _encode_string(field_number: int, text: str) -> bytes:
    return _encode_length_delimited(field_number, text.encode("utf-8"))


def _encode_int64_field(field_number: int, value: int) -> bytes:
    return _encode_tag(field_number, 0) + _encode_varint(value)


def _encode_tensor_proto_double(name: str, array: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(array, dtype="<f8")
    msg = bytearray()
    for d in arr.shape:
        msg += _encode_int64_field(1, int(d))
    msg += _encode_int64_field(2, 11)
    msg += _encode_string(8, name)
    msg += _encode_length_delimited(9, arr.tobytes())
    return bytes(msg)


def _encode_node_proto(
    op_type: str,
    inputs: List[str],
    outputs: List[str],
    name: str,
) -> bytes:
    msg = bytearray()
    for inp in inputs:
        msg += _encode_string(1, inp)
    for out in outputs:
        msg += _encode_string(2, out)
    msg += _encode_string(3, name)
    msg += _encode_string(4, op_type)
    return bytes(msg)


def export_minimal_onnx_proto(net: R2D2Network, onnx_path: Path) -> str:
    """Write a self-contained ONNX ModelProto containing the R2D2 graph and initializers."""
    state = net.get_state_dict()
    schema_ver = infer_schema_version_for_input_dim(net.input_dim)
    order = tensor_order_for_dims(net.input_dim, net.hidden_dim, net.num_actions, net.num_qubits)
    graph_msg = bytearray()

    nodes = [
        _encode_node_proto("Mul", ["obs_raw", "feature_scale"], ["obs_scaled"], "FeatScale"),
        _encode_node_proto("Add", ["obs_scaled", "feature_offset"], ["obs_norm"], "FeatOffset"),
        _encode_node_proto(
            "LSTM",
            ["obs_norm", "W_ih", "W_hh", "b_ih", "h_prev", "c_prev"],
            ["h_next", "c_next"],
            "RecurrentLSTMEncoder",
        ),
        _encode_node_proto("Gemm", ["h_next", "W_val", "b_val"], ["state_value"], "ValueStream"),
        _encode_node_proto("Gemm", ["h_next", "W_adv", "b_adv"], ["advantages"], "AdvantageStream"),
        _encode_node_proto("ReduceMean", ["advantages"], ["adv_mean"], "AdvantageMean"),
        _encode_node_proto("Sub", ["advantages", "adv_mean"], ["adv_centered"], "AdvantageCenter"),
        _encode_node_proto("Add", ["state_value", "adv_centered"], ["q_values"], "DuelingCombine"),
        _encode_node_proto("Gemm", ["h_next", "W_gain", "b_gain"], ["gain_pre"], "GainLinear"),
        _encode_node_proto("Tanh", ["gain_pre"], ["qubit_gains_unit"], "GainTanh"),
    ]
    for n_bytes in nodes:
        graph_msg += _encode_length_delimited(1, n_bytes)

    graph_name = (
        "AVID_R2D2_Dueling_LSTM_V2_Graph"
        if schema_ver == 2
        else "AVID_R2D2_Dueling_LSTM_Graph"
    )
    graph_msg += _encode_string(2, graph_name)

    for name, expected_shape in order:
        arr = state[name]
        if arr.shape != expected_shape:
            raise ValueError(f"ONNX initializer '{name}' shape {arr.shape} != {expected_shape}")
        t_bytes = _encode_tensor_proto_double(name, arr)
        graph_msg += _encode_length_delimited(5, t_bytes)

    opset_msg = _encode_string(1, "") + _encode_int64_field(2, 13)
    model_msg = bytearray()
    model_msg += _encode_int64_field(1, 8)
    model_msg += _encode_string(2, "AVID_R2D2_Exporter")
    model_msg += _encode_string(3, f"{schema_ver}.0.0")
    model_msg += _encode_string(4, "ai.avid.qec")
    model_msg += _encode_int64_field(5, schema_ver)
    model_msg += _encode_length_delimited(7, bytes(graph_msg))
    model_msg += _encode_length_delimited(8, opset_msg)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_bytes = bytes(model_msg)
    onnx_path.write_bytes(onnx_bytes)
    return hashlib.sha256(onnx_bytes).hexdigest()


def export_r2d2_checkpoint(
    net: R2D2Network,
    bin_path: Path,
    json_path: Path,
    onnx_path: Optional[Path] = None,
    training_metrics: Optional[Dict[str, object]] = None,
    schema_version: Optional[int] = None,
) -> Dict[str, object]:
    """Export trained R2D2 weights to binary (.bin), metadata (.json), and ONNX (.onnx)."""
    bin_path = Path(bin_path)
    json_path = Path(json_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    ver = (
        int(schema_version)
        if schema_version is not None
        else infer_schema_version_for_input_dim(net.input_dim)
    )
    if (ver == 2 and net.input_dim != OBS_V2_DIM) or (ver == 1 and net.input_dim != OBS_V1_DIM):
        raise ValueError(
            f"Incompatible schema_version={ver} for network input_dim={net.input_dim}"
        )

    order = tensor_order_for_dims(net.input_dim, net.hidden_dim, net.num_actions, net.num_qubits)
    state = net.get_state_dict()

    header = struct.pack(
        "<8sIIIIII",
        b"AVIDR2D2",
        ver,
        net.input_dim,
        net.hidden_dim,
        net.num_actions,
        net.num_qubits,
        0,
    )
    assert len(header) == 32

    payload_parts: List[bytes] = [header]
    tensor_stats: Dict[str, Dict[str, object]] = {}
    total_floats = 0

    for name, expected_shape in order:
        arr = np.ascontiguousarray(state[name], dtype="<f8")
        if arr.shape != expected_shape:
            raise ValueError(f"Tensor '{name}' shape {arr.shape} != {expected_shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in tensor '{name}'")
        payload_parts.append(arr.tobytes())
        total_floats += int(arr.size)
        tensor_stats[name] = {
            "shape": list(arr.shape),
            "num_elements": int(arr.size),
            "l2_norm": float(np.linalg.norm(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    raw_bytes = b"".join(payload_parts)
    bin_path.write_bytes(raw_bytes)
    bin_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    onnx_sha256 = None
    if onnx_path is not None:
        onnx_sha256 = export_minimal_onnx_proto(net, Path(onnx_path))

    meta: Dict[str, object] = {
        "model_name": f"AVID_R2D2_Dueling_LSTM_V{ver}",
        "schema_version": ver,
        "magic": "AVIDR2D2",
        "architecture": {
            "input_dim": net.input_dim,
            "hidden_dim": net.hidden_dim,
            "num_actions": net.num_actions,
            "num_qubits": net.num_qubits,
            "gain_clip": net.gain_clip,
            "trainable_parameters": net.num_trainable_parameters,
            "normalization_elements": 2 * net.input_dim,
        },
        "binary_checkpoint": {
            "path": str(bin_path.name),
            "size_bytes": len(raw_bytes),
            "total_float64_elements": total_floats,
            "sha256": bin_sha256,
        },
        "onnx_checkpoint": {
            "path": str(Path(onnx_path).name) if onnx_path else None,
            "sha256": onnx_sha256,
        },
        "tensor_stats": tensor_stats,
        "training_metrics": training_metrics or {},
    }
    if ver == 2:
        meta["feature_contract"] = {
            "contract_name": "AVID_Layer2_Observation_Contract_V2",
            "schema_version": 2,
            "num_features": OBS_V2_DIM,
            "feature_names": list(FEATURE_NAMES_82),
        }

    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_r2d2_checkpoint_bin(
    bin_path: Path,
    expected_version: Optional[int] = None,
) -> Tuple[R2D2Network, Dict[str, np.ndarray]]:
    """Load and validate an R2D2Network from a V1 or V2 binary (.bin) checkpoint file."""
    raw = Path(bin_path).read_bytes()
    if len(raw) < 32:
        raise ValueError(f"Checkpoint file too small: {len(raw)} bytes")

    magic, version, in_dim, hid_dim, n_act, n_qub, _ = struct.unpack("<8sIIIIII", raw[:32])
    if magic != b"AVIDR2D2":
        raise ValueError(f"Invalid checkpoint magic: {magic!r}")
    if version not in (1, 2):
        raise ValueError(f"Unsupported checkpoint version: {version}")
    if expected_version is not None and int(version) != int(expected_version):
        raise ValueError(
            f"Checkpoint version mismatch: expected V{expected_version}, got V{version}"
        )
    if version == 1 and in_dim != OBS_V1_DIM:
        raise ValueError(f"V1 checkpoint must have input_dim=8, got {in_dim}")
    if version == 2 and in_dim != OBS_V2_DIM:
        raise ValueError(f"V2 checkpoint must have input_dim=82, got {in_dim}")

    order = tensor_order_for_dims(in_dim, hid_dim, n_act, n_qub)
    net = R2D2Network(
        input_dim=in_dim,
        hidden_dim=hid_dim,
        num_actions=n_act,
        num_qubits=n_qub,
    )

    offset = 32
    state_dict: Dict[str, np.ndarray] = {}
    for name, shape in order:
        count = int(np.prod(shape))
        byte_len = count * 8
        if offset + byte_len > len(raw):
            raise ValueError(f"Truncated checkpoint while reading '{name}'")
        arr = np.frombuffer(raw, dtype="<f8", count=count, offset=offset).reshape(shape).copy()
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in checkpoint tensor '{name}'")
        state_dict[name] = arr
        offset += byte_len

    if offset != len(raw):
        raise ValueError(f"Extra trailing bytes in checkpoint: {len(raw) - offset}")

    net.load_state_dict(state_dict)
    return net, state_dict
