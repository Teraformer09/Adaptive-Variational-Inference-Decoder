import numpy as np
from avid_model import *
th1 = theta_uniform(0.01)
th2 = th1.copy(); th2[idx_edge(4,5)] = 0.08
th3 = th1.copy(); th3[idx_single(3)] = 0.08
# --- sanity: DP vs brute force
for name, th in [("sim1",th1),("sim2",th2),("sim3",th3)]:
    d = np.abs(dp_joint(th) - brute_joint(th)).max()
    print(f"{name}: max |DP - brute force| = {d:.2e}")
# --- structure checks
print("signatures per mechanism (single 0..8):", [hex(SIG[q]) for q in range(9)])
print("Z3 sig", hex(SIG[3]), "Z6 sig", hex(SIG[6]), "Z4Z5", hex(SIG[idx_edge(4,5)]), "Z7Z8", hex(SIG[idx_edge(7,8)]))
# --- Dhaya's table
P = {k: dp_joint(t) for k,t in [(1,th1),(2,th2),(3,th3)]}
opt = {k: map_decision(P[k]) for k in P}
print("\nLER_opt (%):", {k: round(100*ler_opt(P[k]),2) for k in P})
dec_always0 = np.zeros(16, int)
# coin flip at the 8 syndromes tied in Sim-1
tied = [s for s in range(16) if abs(P[1][s,0]-P[1][s,1])<1e-12]
print("tied syndromes in sim1:", [hex(s) for s in tied])
# static proxy: optimal for independent single-qubit faults only (p=0.01)
th_ind = np.zeros(M)+1e-15; th_ind[:9] = 0.01
Pind = dp_joint(th_ind); dec_static = map_decision(Pind)
rows = []
for k in (1,2,3):
    row = [100*ler_opt(P[k]), 100*ler_of(P[k], opt[2]), 100*ler_of(P[k], opt[3]), 100*ler_of(P[k], dec_always0),
           100*sum(0.5*(P[k][s,0]+P[k][s,1]) if s in tied else min(P[k][s]) for s in range(16)),
           100*ler_of(P[k], dec_static)]
    rows.append(row)
print("\nTruth | Optimal | Tuned Z4Z5 | Tuned Z3 | Always0 | Coin@tied | Static proxy")
for k,r in zip((1,2,3),rows): print(k, "  ", "  ".join(f"{x:6.2f}" for x in r))
print("\nDhaya's table: 1: 5.71 5.71 5.71 5.71 5.71 5.74 | 2: 5.71 5.71 12.04 5.71 8.87 11.82 | 3: 5.71 12.04 5.71 12.04 8.87 5.99")
# --- 0x4 share of extra error for tuned-to-wrong
extra = [P[3][s,1-opt[2][s]]-min(P[3][s]) for s in range(16)]
print("share of extra error at 0x4 (truth 3, decoder tuned to 2): %.1f%%" % (100*extra[4]/sum(extra)))
