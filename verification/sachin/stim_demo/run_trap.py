import numpy as np, csv
from avid_model import *; from avid_decoders import *; from avid_stim import *
PRIOR = theta_prior()
# Same histogram for every x: orbit 0x4 holds Z3, Z6, Z4Z5, Z7Z8; only prod(1-2*theta_k) matters for P(s).
base = PRIOR.copy(); base[idx_single(4)] = 0.055
PROD = (1-2*0.012)*(1-2*0.075)
a_lo, a_hi = 0.012, (1-PROD/(1-2*0.004))/2
rows=[]; W=4096; NSEED=40
for x in np.linspace(0,1,11):             # x = fraction of the excess moved from pair45 to Z3
    a = a_lo + x*(a_hi-a_lo); b = (1-PROD/(1-2*a))/2
    th = base.copy(); th[idx_single(3)] = a; th[idx_edge(4,5)] = b
    Pt = dp_joint(th)
    sampler = build_one_round(th).compile_detector_sampler(seed=77)
    tabs = {"B0 static": minweight_table(marginal_llr_q78(PRIOR)),
            "oracle": map_decision(Pt)}
    res = {k: [] for k in ["B0 static","AVID projection (MAP+proj, no RL)","MAP + exact pair-aware","oracle"]}
    for seed in range(NSEED):
        det,_ = sampler.sample(W, separate_observables=True)
        h = np.bincount((det.astype(np.uint8)<<np.arange(4)).sum(1).astype(int), minlength=16).astype(float)
        thh = fit_map(h)
        res["B0 static"].append(ler_of(Pt, tabs["B0 static"]))
        res["AVID projection (MAP+proj, no RL)"].append(ler_of(Pt, minweight_table(projected_llr_q78(thh, PRIOR))))
        res["MAP + exact pair-aware"].append(ler_of(Pt, joint_table(thh)))
        res["oracle"].append(ler_of(Pt, tabs["oracle"]))
    hist_dev = np.abs(p_syndrome(th) - p_syndrome(base.copy()*0+ (lambda t:(t.__setitem__(idx_single(3),a_lo) or t.__setitem__(idx_edge(4,5),0.075) or t))(base.copy()))).max()
    p_l0 = Pt[4,0]/Pt[4].sum()
    rows.append([x,a,b,p_l0,hist_dev]+[np.mean(res[k]) for k in res])
    print(f"x={x:.1f} Z3={a:.3f} pair45={b:.3f} P(l=0|0x4)={p_l0:.2f} histdev={hist_dev:.1e} | "+"  ".join(f"{k.split(' (')[0][:14]}={100*np.mean(v):5.2f}" for k,v in res.items()))
np.savetxt("trap_curve.csv", np.array(rows), delimiter=",", header="x,theta_Z3,theta_pair45,P_l0_given_0x4,max_hist_dev,B0_static,AVID_proj,MAP_exact,oracle", comments="")

# Stim-sampled empirical check of the exact LERs (independent of the DP): W=4096, seed 0 tables
print("\nEmpirical (Stim failure counts, 2e6 shots) vs exact:")
for name, th in [("R1 repo correlated", PRIOR.copy()), ("R3 alias-swap", PRIOR.copy())]:
    th[idx_single(4)] = 0.055
    if name.startswith("R1"): th[idx_edge(4,5)] = 0.075
    else: th[idx_single(3)] = 0.075
    Pt = dp_joint(th); sampler = build_one_round(th).compile_detector_sampler(seed=5)
    det,_ = sampler.sample(4096, separate_observables=True)
    h = np.bincount((det.astype(np.uint8)<<np.arange(4)).sum(1).astype(int), minlength=16).astype(float)
    thh = fit_map(h)
    tabs = {"B0 static": minweight_table(marginal_llr_q78(PRIOR)), "AVID proj": minweight_table(projected_llr_q78(thh,PRIOR)), "oracle": map_decision(Pt)}
    s, l = sample_syn_ell(th, 2_000_000, seed=99)
    for k,t in tabs.items():
        fails = int((t[s] != l).sum()); p = fails/len(s); se = (p*(1-p)/len(s))**.5
        print(f"  {name:20s} {k:10s} failures={fails:7d}  LER={100*p:.3f}% +/- {100*1.96*se:.3f}   exact={100*ler_of(Pt,t):.3f}%")
