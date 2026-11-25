"""Stim circuits for the d=3 Z-error model (independent sampler, not the team's C++ generator)."""
import stim, numpy as np
from avid_model import *

def _checks_str():
    return " ".join("*".join(f"X{q}" for q in sup) for sup in XCHECKS)

def build_one_round(theta):
    """Code-capacity circuit: |+>^9, Z / ZZ events with probabilities theta, ideal X-check readout.
    detectors = 4 X-check outcomes (bit c = check c); observable 0 = logical bit l (X0*X3*X6)."""
    c = stim.Circuit()
    c.append("RX", list(range(N_Q)))
    for k in range(M):
        qs = SUPPORT[k]
        if len(qs) == 1:
            c.append("Z_ERROR", [qs[0]], float(theta[k]))
        else:
            c.append("CORRELATED_ERROR", [stim.target_z(qs[0]), stim.target_z(qs[1])], float(theta[k]))
    c.append_from_stim_program_text("MPP " + _checks_str())
    for i in range(4):
        c.append("DETECTOR", [stim.target_rec(-4 + i)])
    c.append_from_stim_program_text("MPP X0*X3*X6")
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return c

def sample_syn_ell(theta, shots, seed):
    c = build_one_round(theta)
    det, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    s = (det.astype(np.uint8) << np.arange(4)).sum(1)
    return s.astype(int), obs[:,0].astype(int)

def hist_joint(s, l):
    H = np.zeros((16,2)); np.add.at(H, (s,l), 1); return H
