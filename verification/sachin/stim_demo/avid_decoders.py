"""Estimator (exact 16-state joint-DP MAP) and decoders. Projection is a line-by-line Python port of
NoiseModel::to_pair_projected_q7_8_llr (software/src/inference/noise_model.cpp, branch AVID_EPJ_Patches,
no per-qubit gains, pair_weight_scale=1.0). Decoding from LLRs = exact minimum-total-LLR pattern over all
512 Z-patterns (the objective BP+OSD-0 targets); this is NOT a bit-exact model of the RTL BP/OSD."""
import numpy as np, math
from scipy.optimize import minimize
from avid_model import *

# ---------- 16-state likelihood with analytic gradient (Walsh-Hadamard form) ----------
_U = np.arange(16)
PAR = np.array([[bin(u & int(SIG[k])).count("1") & 1 for k in range(M)] for u in _U])   # (16,21)
WHT = np.array([[(-1)**(bin(s & u).count("1") & 1) for u in _U] for s in _U], float)   # (16,16)

def p_syndrome(theta):
    A = 1.0 - 2.0*theta[None,:]*PAR
    return (WHT @ np.prod(A, axis=1)) / 16.0

def nll_and_grad(theta, phat, theta0, lam):
    A = 1.0 - 2.0*theta[None,:]*PAR
    Ph = np.prod(A, axis=1)
    P = np.maximum((WHT @ Ph)/16.0, 1e-300)
    nll = -(phat*np.log(P)).sum() + 0.5*lam*((theta-theta0)**2).sum()
    dPh = (-2.0*PAR) * (Ph[:,None]/A)                     # (16,21)
    dP = (WHT @ dPh)/16.0                                  # (16,21)
    grad = -(phat/P) @ dP + lam*(theta-theta0)
    return nll, grad

def fit_map(hist16, theta0=None, lam=0.05, maxiter=200):
    theta0 = theta_prior() if theta0 is None else theta0
    phat = hist16/hist16.sum()
    r = minimize(nll_and_grad, theta0.copy(), args=(phat, theta0, lam), jac=True, method="L-BFGS-B",
                 bounds=[(1e-5,0.45)]*M, options={"maxiter":maxiter})
    return r.x

# ---------- LLR constructions ----------
def effective_p(theta):
    p = np.clip(theta[:N_Q], 1e-6, 0.49).copy()
    for k,(i,j) in enumerate(EDGES):
        pe = np.clip(theta[N_Q+k], 1e-6, 0.49)
        for v in (i,j):
            p[v] = p[v]*(1-pe) + (1-p[v])*pe
    return np.clip(p, 1e-5, 0.49)

def _lround(x): return int(math.floor(abs(x)+0.5))*(1 if x>=0 else -1)

def marginal_llr_q78(theta):
    p = effective_p(theta)
    return np.array([min(max(_lround(math.log((1-pp)/pp)*256.0),16),0x3FFF) for pp in p])

def projected_llr_q78(theta, prior_theta, pws=1.0):
    llr = marginal_llr_q78(theta).astype(int)
    if pws <= 0: return llr
    for g in range(1,16):
        best_k, best_score = -1, -1.0
        for k,(u,v) in enumerate(EDGES):
            if SIG[N_Q+k]==g and SIG[u]!=SIG[v]:
                score = theta[N_Q+k] + 0.25*(theta[u]+theta[v])
                if score > best_score: best_score, best_k = score, k
        if best_k < 0: continue
        aliased = [w for w in range(N_Q) if SIG[w]==g]
        if not aliased: continue
        u,v = EDGES[best_k]
        p_pair = theta[N_Q+best_k]; p_end_max = max(theta[u],theta[v])
        p_base_prior = min(0.015, prior_theta[aliased[0]])
        dominated = False
        for k2,(u2,v2) in enumerate(EDGES):
            if k2==best_k: continue
            if (u2 in (u,v) or v2 in (u,v)) and theta[N_Q+k2] > p_pair + 0.008:
                dominated = True; break
        if dominated: continue
        gain_boost = 1.0
        end_support = 0.48*max(0.0,p_end_max-p_base_prior) if p_pair>0.012 else 0.0
        pair_evidence = pws*gain_boost*(p_pair+end_support)
        if pair_evidence > p_base_prior + 0.008:
            p_eff_pair = min(max(max(p_pair,0.65*p_end_max),0.015),0.38)
            eff_scale = 0.85 + 0.15*pws + 0.85*(gain_boost-1.0)
            scale_inv = 1.0/min(max(eff_scale,0.50),2.20)
            half = min(max(_lround(0.48*scale_inv*math.log((1-p_eff_pair)/p_eff_pair)*256.0),16),0x200)
            llr[u] = min(llr[u], half); llr[v] = min(llr[v], half)
            pair_sum = llr[u]+llr[v]
            for w in aliased:
                llr[w] = max(llr[w], pair_sum + 0x20)
    return llr

# ---------- decoders as 16-entry tables: predicted logical bit per syndrome ----------
_PAT = np.arange(512)
_PAT_BITS = ((_PAT[:,None] >> np.arange(N_Q)) & 1)                                  # (512,9)
_PAT_SYN = np.zeros(512, int); _PAT_ELL = np.zeros(512, int)
for q in range(N_Q):
    _PAT_SYN ^= _PAT_BITS[:,q]*qubit_sig(q)
    _PAT_ELL ^= _PAT_BITS[:,q]*int(q in LOGICAL_SUPPORT)

def minweight_table(llr):
    """argmin_{pattern with syndrome s} sum(llr over flipped qubits); ties -> lowest pattern index."""
    w = _PAT_BITS @ np.asarray(llr, float)
    table = np.zeros(16, int)
    for s in range(16):
        cand = np.where(_PAT_SYN==s)[0]
        table[s] = _PAT_ELL[cand[np.argmin(w[cand])]]
    return table

def joint_table(theta):          # exact pair-aware MAP bit given a parameter vector
    return map_decision(dp_joint(theta))

def matching_table(theta):       # PyMatching on the marginal decoding graph (independent implementation)
    import pymatching
    p = effective_p(theta); m = pymatching.Matching()
    for q in range(N_Q):
        w = math.log((1-p[q])/p[q]); sig = qubit_sig(q)
        cs = [c for c in range(4) if (sig>>c)&1]
        fid = {0} if q in LOGICAL_SUPPORT else set()
        if len(cs)==2: m.add_edge(cs[0], cs[1], fault_ids=fid, weight=w, merge_strategy="smallest-weight")
        else:          m.add_boundary_edge(cs[0], fault_ids=fid, weight=w, merge_strategy="smallest-weight")
    out = np.zeros(16, int)
    for s in range(16):
        det = np.array([(s>>c)&1 for c in range(4)], np.uint8)
        out[s] = int(m.decode(det)[0])
    return out
