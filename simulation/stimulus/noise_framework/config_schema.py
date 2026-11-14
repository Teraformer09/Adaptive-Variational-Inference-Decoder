"""Data contracts and configuration schema validation for the AVID Seven-Stage Noise Framework.

Enforces:
  1. `NoiseProfile` (frozen dataclass for each of the 16 PDs).
  2. `GeneratorSpec` (frozen dataclass for active generator instances).
  3. `NoiseContribution` (provenance-tagged single-generator output at sample_id, timestamp).
  4. `CompositeNoiseSample` (combined physical noise sample preserving all individual contributions).
  5. Exact truncated Gaussian sampling on [min, max] without boundary clipping mass pile-up.
  6. Canonical 21-parameter ordering matching `software/src/inference/noise_model.h`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import yaml


# Canonical 21-parameter ordering in AVID (9 single-qubit + 12 nearest-neighbour pairs)
CANONICAL_21_PARAM_NAMES: Tuple[str, ...] = (
    "p_0",
    "p_1",
    "p_2",
    "p_3",
    "p_4",
    "p_5",
    "p_6",
    "p_7",
    "p_8",
    "p_01",
    "p_12",
    "p_34",
    "p_45",
    "p_67",
    "p_78",
    "p_03",
    "p_36",
    "p_14",
    "p_47",
    "p_25",
    "p_58",
)

# Canonical 12 nearest-neighbour pair edges matching `CANONICAL_21_PARAM_NAMES[9:21]`
CANONICAL_12_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (0, 3),
    (3, 6),
    (1, 4),
    (4, 7),
    (2, 5),
    (5, 8),
)

VALID_DISTRIBUTION_TYPES = {"truncated_gaussian", "gaussian", "beta", "uniform"}
VALID_OUTPUT_DOMAINS = {
    "probability_parameter",
    "pauli_fault_mask",
    "measurement_fault_parameters",
}


def derive_deterministic_seed(master_seed: int, *components: Any) -> int:
    """Derive a deterministic 63-bit integer seed H(master_seed, *components).

    Ensures that adding or reconfiguring one generator never changes the RNG stream
    of any other generator or assignment window.
    """
    payload = f"{int(master_seed)}::" + "::".join(str(c) for c in components)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _std_normal_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal CDF Phi(x) via math.erf / vectorised erf."""
    arr = np.asarray(x, dtype=np.float64)
    erf_vec = np.vectorize(math.erf, otypes=[np.float64])
    res = 0.5 * (1.0 + erf_vec(arr / math.sqrt(2.0)))
    if np.ndim(x) == 0:
        return float(res)
    return res


def _std_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _std_normal_ppf(p: float | np.ndarray) -> np.ndarray:
    """High-accuracy Acklam rational approximation to the standard normal inverse CDF Phi^{-1}(p)."""
    u = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    a = np.array([
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ])
    b = np.array([
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ])
    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ])
    d = np.array([
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ])
    p_low = 0.02425
    p_high = 1.0 - p_low

    out = np.empty_like(u, dtype=np.float64)
    low_mask = u < p_low
    high_mask = u > p_high
    mid_mask = (~low_mask) & (~high_mask)

    if np.any(low_mask):
        q = np.sqrt(-2.0 * np.log(u[low_mask]))
        out[low_mask] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(mid_mask):
        q = u[mid_mask] - 0.5
        r = q * q
        out[mid_mask] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    if np.any(high_mask):
        q = np.sqrt(-2.0 * np.log(1.0 - u[high_mask]))
        out[high_mask] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    return out


def sample_truncated_gaussian(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int | Tuple[int, ...] = 1,
) -> np.ndarray:
    """Sample from the exact truncated normal distribution N(mean, std^2) restricted to [low, high].

    Uses inverse-CDF sampling (not post-hoc boundary clipping) so there is zero artificial
    point-mass accumulation at `low` or `high`.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}")
    if not (low < high):
        raise ValueError(f"Invalid support [{low}, {high}]")
    alpha = (low - mean) / std
    beta = (high - mean) / std
    cdf_alpha = float(_std_normal_cdf(alpha))
    cdf_beta = float(_std_normal_cdf(beta))
    span = max(cdf_beta - cdf_alpha, 1e-12)
    u = rng.uniform(cdf_alpha, cdf_alpha + span, size=size)
    z = _std_normal_ppf(u)
    samples = mean + std * z
    return np.clip(samples, low, high)


def theoretical_truncated_gaussian_moments(
    mean: float, std: float, low: float, high: float
) -> Tuple[float, float]:
    """Compute exact analytical (mean, variance) of N(mean, std^2) truncated to [low, high]."""
    alpha = (low - mean) / std
    beta = (high - mean) / std
    z_norm = max(float(_std_normal_cdf(beta)) - float(_std_normal_cdf(alpha)), 1e-12)
    phi_alpha = _std_normal_pdf(alpha)
    phi_beta = _std_normal_pdf(beta)
    ratio = (phi_alpha - phi_beta) / z_norm
    exact_mean = mean + std * ratio
    exact_var = (std * std) * (
        1.0 + (alpha * phi_alpha - beta * phi_beta) / z_norm - ratio * ratio
    )
    return float(exact_mean), float(max(exact_var, 1e-16))


@dataclass(frozen=True)
class NoiseProfile:
    """Immutable representation of one of the 16 Probability Distributions (PD_01 .. PD_16)."""

    profile_id: str
    distribution_type: str
    parameters: Dict[str, Any]
    support: Dict[str, Any]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.profile_id or not isinstance(self.profile_id, str):
            raise ValueError("profile_id must be a non-empty string")
        if self.distribution_type not in VALID_DISTRIBUTION_TYPES:
            raise ValueError(
                f"Unsupported distribution_type '{self.distribution_type}' in {self.profile_id}"
            )
        if not isinstance(self.parameters, dict) or "mean" not in self.parameters or "std" not in self.parameters:
            raise ValueError(f"Profile {self.profile_id} missing required 'mean'/'std' parameters")
        mean = float(self.parameters["mean"])
        std = float(self.parameters["std"])
        if not (math.isfinite(mean) and math.isfinite(std) and std > 0.0):
            raise ValueError(f"Profile {self.profile_id} has invalid mean={mean} or std={std}")
        if not isinstance(self.support, dict) or "min" not in self.support or "max" not in self.support:
            raise ValueError(f"Profile {self.profile_id} missing required support bounds")
        s_min = float(self.support["min"])
        s_max = float(self.support["max"])
        if not (0.0 <= s_min < s_max <= 1.0):
            raise ValueError(
                f"Profile {self.profile_id} support [{s_min}, {s_max}] violates probability interval [0, 1]"
            )
        if not (s_min <= mean <= s_max):
            raise ValueError(
                f"Profile {self.profile_id} mean {mean} lies outside support [{s_min}, {s_max}]"
            )
        if "variable" not in self.support or "units" not in self.support:
            raise ValueError(f"Profile {self.profile_id} support must declare 'variable' and 'units'")

    @property
    def min_val(self) -> float:
        return float(self.support["min"])

    @property
    def max_val(self) -> float:
        return float(self.support["max"])

    @property
    def mean(self) -> float:
        return float(self.parameters["mean"])

    @property
    def std(self) -> float:
        return float(self.parameters["std"])

    def sample_values(
        self, rng: np.random.Generator, size: int | Tuple[int, ...] = 1
    ) -> np.ndarray:
        """Draw exact truncated samples within `[support.min, support.max]`."""
        return sample_truncated_gaussian(
            rng=rng,
            mean=self.mean,
            std=self.std,
            low=self.min_val,
            high=self.max_val,
            size=size,
        )

    def theoretical_moments(self) -> Tuple[float, float]:
        return theoretical_truncated_gaussian_moments(
            mean=self.mean, std=self.std, low=self.min_val, high=self.max_val
        )


@dataclass(frozen=True)
class GeneratorSpec:
    """Immutable specification of an active noise-generator instance."""

    generator_id: str
    generator_type: str
    enabled: bool = True
    seed: int = 42
    output_domain: str = "measurement_fault_parameters"
    output_shape: Tuple[int, ...] = (11,)
    composition_group: str = "measurement_parameters"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.generator_id or not isinstance(self.generator_id, str):
            raise ValueError("generator_id must be a non-empty string")
        if self.output_domain not in VALID_OUTPUT_DOMAINS:
            raise ValueError(
                f"Generator '{self.generator_id}' has unsupported output_domain '{self.output_domain}'"
            )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"Generator '{self.generator_id}' must have a non-negative integer seed")
        if not self.output_shape or any(int(d) <= 0 for d in self.output_shape):
            raise ValueError(f"Generator '{self.generator_id}' has invalid output_shape {self.output_shape}")
        if not self.composition_group:
            raise ValueError(f"Generator '{self.generator_id}' must specify composition_group")


@dataclass
class NoiseContribution:
    """Single-generator output at a synchronized sampling instant with complete provenance."""

    generator_id: str
    profile_id: str
    window_id: int
    sample_id: int
    timestamp: float
    output_domain: str
    values: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.generator_id:
            raise ValueError("NoiseContribution requires generator_id")
        if not self.profile_id:
            raise ValueError("NoiseContribution requires profile_id")
        if self.window_id < 0 or self.sample_id < 0:
            raise ValueError("window_id and sample_id must be non-negative")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"Invalid timestamp {self.timestamp}")
        if self.output_domain not in VALID_OUTPUT_DOMAINS:
            raise ValueError(f"Invalid output_domain {self.output_domain}")
        self.values = np.asarray(self.values)
        if not np.all(np.isfinite(self.values)):
            raise ValueError(f"Non-finite values in contribution from {self.generator_id}")


@dataclass
class CompositeNoiseSample:
    """Combined physical noise realization at a synchronized sampling instant.

    Preserves all individual `NoiseContribution` objects for attribution and validation.
    """

    sample_id: int
    timestamp: float
    window_id: int
    contributions: List[NoiseContribution]
    composite_parameters: Dict[str, Any]
    composite_faults: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_id < 0 or self.window_id < 0:
            raise ValueError("sample_id and window_id must be non-negative")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"Invalid timestamp {self.timestamp}")
        if not isinstance(self.contributions, list) or len(self.contributions) == 0:
            raise ValueError("CompositeNoiseSample must retain non-empty individual contributions")


def load_yaml_validated(path: str | Path, required_keys: Sequence[str] = ()) -> Dict[str, Any]:
    """Load a YAML file and verify required top-level keys."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root in {p} must be a mapping")
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing required key '{k}' in {p}")
    return data


def compute_canonical_config_hash(obj: Any) -> str:
    """Compute a deterministic SHA-256 hex digest of a configuration structure."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
