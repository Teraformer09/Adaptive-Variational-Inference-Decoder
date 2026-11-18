"""Integration Test: Hardware Control Register & Q7.8 Weight Bank Updates (`test_decoder_control_update.py`)."""

from __future__ import annotations

import numpy as np

from training.environments.decoder_in_loop import (
    STATIC_LUT_Z,
    count_logical_z_failures,
    decode_window_minsum_osd0_z,
    project_q78_weights_d3,
    sample_d3_circuit_window,
)
from training.models.r2d2_network import ACTION_DEFINITIONS


def test_hardware_q78_weight_projection_and_decoder_control_effectiveness() -> None:
    """Verify that pair_weight_scale, bp_iterations, qubit_gains, and fallback produce valid Q7.8 banks and reduce ZZ LER."""
    rng = np.random.default_rng(202699)
    theta_z = np.full(21, 0.011, dtype=np.float64)
    theta_z[4] = 0.042
    theta_z[5] = 0.038
    theta_z[9 + 3] = 0.092  # Strong ZZ pair (4, 5) -> X-syndrome 0x4 (aliases with single qubit q3/q6!)
    theta_x = np.full(21, 0.005, dtype=np.float64)

    circ = sample_d3_circuit_window(rng, theta_z, theta_x, window_rounds=2048)
    err_z = circ["err_z"]
    sx = circ["sx"]

    # 1. Static uniform baseline (action 0)
    f_static = count_logical_z_failures(err_z, STATIC_LUT_Z[sx])

    # 2. Adaptive pair-projected + targeted gain residual (action 2: AGGRESSIVE_PAIR_K5)
    spec_pair = ACTION_DEFINITIONS[2]
    gains_mult = np.ones(9, dtype=np.float64)
    gains_mult[4] = 0.80  # attenuate endpoint LLRs
    gains_mult[5] = 0.80
    w_pair = project_q78_weights_d3(
        qubit_p=theta_z[:9],
        alphas=theta_z[9:],
        pair_weight_scale=float(spec_pair["pair_weight_scale"]),
        qubit_gains=gains_mult,
        fallback=False,
    )
    assert w_pair.shape == (9,)
    assert np.all((w_pair >= 16) & (w_pair <= 0x3FFF))

    corr_pair = decode_window_minsum_osd0_z(sx, w_pair, bp_iterations=int(spec_pair["bp_iterations"]))
    f_pair = count_logical_z_failures(err_z, corr_pair)
    assert f_pair < f_static * 0.75

    # 3. Conservative fallback (action 5) resets all 9 Q7.8 registers to 0x0468
    w_fb = project_q78_weights_d3(
        qubit_p=theta_z[:9],
        alphas=theta_z[9:],
        pair_weight_scale=0.0,
        fallback=True,
    )
    assert np.all(w_fb == 0x0468)
