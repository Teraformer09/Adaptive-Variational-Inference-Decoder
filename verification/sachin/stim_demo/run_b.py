import numpy as np, csv, sys, time
from avid_model import *; from avid_decoders import *; from avid_stim import *
PRIOR = theta_prior()
def sc(name, **kw):
    t = PRIOR.copy()
    for k,v in kw.items():
        if k.startswith("q"): t[idx_single(int(k[1:]))] = v
        else: a,b = k[1:].split("_"); t[idx_edge(int(a),int(b))] = v
    return name, t
def uni(p, **kw):
    t = theta_uniform(p)
    for k,v in kw.items():
        if k.startswith("q"): t[idx_single(int(k[1:]))] = v
        else: a,b = k[1:].split("_"); t[idx_edge(int(a),int(b))] = v
    return t
SCEN = [("D1 uniform 1% (Dhaya sim1)", uni(0.01)),
        ("D2 Z4Z5 burst (Dhaya sim2)", uni(0.01, e4_5=0.08)),
        ("D3 Z3 burst (Dhaya sim3)", uni(0.01, q3=0.08)),
        sc("R0 repo stationary"),
        sc("R1 repo correlated (Z4 + pair45 hot)", q4=0.055, e4_5=0.075),
        sc("R2 repo burst level", q4=0.075, e4_5=0.085),
        sc("R3 alias-swap: Z3 hot instead of pair45", q4=0.055, q3=0.075),
        sc("R4 alias-swap burst level", q4=0.075, q3=0.085)]
WINS = [256, 1024, 4096, 16384]; NSEED = 40
rows = []; t0 = time.time()
for si,(name,th) in enumerate(SCEN):
    Pt = dp_joint(th); dets = {}
    dets["oracle (exact joint, true theta)"] = map_decision(Pt)
    dets["MWPM, true marginals (PyMatching)"] = matching_table(th)
    dets["projection on TRUE theta (no estimation)"] = minweight_table(projected_llr_q78(th, PRIOR))
    dets["static prior, marginal LLR (B0)"] = minweight_table(marginal_llr_q78(PRIOR))
    dets["static prior, exact joint"] = joint_table(PRIOR)
    sampler = build_one_round(th).compile_detector_sampler(seed=1000+si)
    for W in WINS:
        for seed in range(NSEED):
            det, _ = sampler.sample(W, separate_observables=True)
            s = (det.astype(np.uint8) << np.arange(4)).sum(1).astype(int)
            h = np.bincount(s, minlength=16).astype(float)
            thh = fit_map(h)
            d = dict(dets)
            d["MAP + marginal LLR (pure 9-LLR)"]   = minweight_table(marginal_llr_q78(thh))
            d["MAP + AVID projection (no RL)"]     = minweight_table(projected_llr_q78(thh, PRIOR))
            d["MAP + exact pair-aware (ideal proj.)"] = joint_table(thh)
            for k,tab in d.items():
                rows.append((name, W, seed, k, ler_of(Pt, tab)))
    print(f"[{time.time()-t0:5.0f}s] done {name}", flush=True)
with open("results_b.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["scenario","window","seed","decoder","ler_exact"]); w.writerows(rows)
