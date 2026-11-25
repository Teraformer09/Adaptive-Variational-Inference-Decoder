"""d=3 rotated surface code, Z-error (phase-flip) model with 9 single-qubit and 12 nearest-neighbour
pair mechanisms. Conventions follow AVID Baseline_V0 math doc (X-check supports) and Dhaya's note
(logical bit = parity of error on {0,3,6})."""
import numpy as np

N_Q = 9
EDGES = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
XCHECKS = [(0,1),(1,2,4,5),(3,4,6,7),(7,8)]          # X-check supports c0..c3 (repo C++ NM_CHECK_CONN layout)
XCHECKS_DOC_LAYOUT = [(0,1,3,4),(1,2),(4,5,7,8),(6,7)]  # Baseline_V0 doc Step 1.1 text (vertically flipped)
LOGICAL_SUPPORT = (0,3,6)                             # l = parity(e & {0,3,6})

def qubit_sig(q):
    return sum(1<<c for c,sup in enumerate(XCHECKS) if q in sup)

def mech_masks():
    """Return (sig[21], ell[21], supports) for 9 singles then 12 edges."""
    sig, ell, sup = [], [], []
    for q in range(N_Q):
        sig.append(qubit_sig(q)); ell.append(int(q in LOGICAL_SUPPORT)); sup.append((q,))
    for (i,j) in EDGES:
        sig.append(qubit_sig(i) ^ qubit_sig(j))
        ell.append(int(i in LOGICAL_SUPPORT) ^ int(j in LOGICAL_SUPPORT)); sup.append((i,j))
    return np.array(sig), np.array(ell), sup

SIG, ELL, SUPPORT = mech_masks()
M = len(SIG)

def dp_joint(theta):
    """Exact P(s,l) as array (16,2) by sequential XOR-convolution over the 21 mechanisms."""
    P = np.zeros((16,2)); P[0,0] = 1.0
    idx_all = np.arange(16)
    for k in range(M):
        s_k, l_k, p = SIG[k], ELL[k], theta[k]
        shifted = np.stack([P[idx_all ^ s_k, 0 ^ l_k], P[idx_all ^ s_k, 1 ^ l_k]], axis=1)
        P = (1-p)*P + p*shifted
    return P

def brute_joint(theta):
    """Independent check: sum P(b) over all 2^21 subsets b of fired mechanisms."""
    n = M
    B = np.arange(1<<n, dtype=np.uint32)
    s = np.zeros(1<<n, dtype=np.uint8); l = np.zeros(1<<n, dtype=np.uint8)
    logp = np.zeros(1<<n)
    for k in range(n):
        bit = ((B>>k)&1).astype(bool)
        s ^= np.where(bit, SIG[k], 0).astype(np.uint8)
        l ^= np.where(bit, ELL[k], 0).astype(np.uint8)
        logp += np.where(bit, np.log(theta[k]), np.log(1-theta[k]))
    P = np.zeros((16,2))
    np.add.at(P, (s.astype(int), l.astype(int)), np.exp(logp))
    return P

def ler_of(P, decision):
    """LER of decision table d[s] in {0,1}: sum_s P(s, 1-d(s))."""
    return float(sum(P[s, 1-decision[s]] for s in range(16)))

def map_decision(P):
    """MAP logical bit per syndrome (ties -> 0)."""
    return np.array([int(P[s,1] > P[s,0]) for s in range(16)])

def ler_opt(P):
    return float(np.minimum(P[:,0], P[:,1]).sum())

def theta_uniform(p=0.01):
    return np.full(M, p)

def idx_single(q): return q
def idx_edge(i,j): return N_Q + EDGES.index((i,j))

# Cross-check against the repo C++ tables (noise_model.cpp NM_EVENT_SIGNATURES)
CPP_SIGS = [0x1,0x3,0x2,0x4,0x6,0x2,0x4,0xC,0x8, 0x2,0x1,0x2,0x4,0x8,0x4,0x5,0x0,0x5,0xA,0x0,0xA]
assert list(SIG) == CPP_SIGS, (list(SIG), CPP_SIGS)

# Static prior used by the repo's static B0 / estimator prior
PRIOR_SINGLE, PRIOR_PAIR = 0.012, 0.004
def theta_prior():
    t = np.full(M, PRIOR_PAIR); t[:N_Q] = PRIOR_SINGLE; return t
