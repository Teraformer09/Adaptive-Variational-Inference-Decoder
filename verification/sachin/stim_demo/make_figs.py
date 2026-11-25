import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
C = {"B0":"#8a8a8a","AVID":"#c0392b","ORA":"#1f6f8b","EX":"#7d6b9d"}
# Fig 1: trap curve
d = np.genfromtxt("trap_curve.csv", delimiter=",", names=True)
fig, ax = plt.subplots(figsize=(6.6,3.9))
ax.plot(d["x"],100*d["B0_static"],color=C["B0"],lw=2,label="Static prior (B0)")
ax.plot(d["x"],100*d["AVID_proj"],color=C["AVID"],lw=2.4,label="MAP + AVID projection (no RL)")
ax.plot(d["x"],100*d["oracle"],color=C["ORA"],lw=2,ls="--",label="Optimal (knows the rates)")
ax.axvline(0.45,color="k",lw=0.8,ls=":"); ax.text(0.47,10.9,"best decision flips here\n(P(l=0 | s=0x4) = 0.5)",fontsize=8,va="top")
ax.set_xlabel("share of the excess syndrome-0x4 mass caused by a single-qubit fault (Z3) instead of the pair Z4Z5")
ax.set_ylabel("logical error rate (%)"); ax.set_ylim(0,11)
ax.set_title("Same 16-bin histogram at every x, different cause",fontsize=10,loc="left")
ax.legend(frameon=False,fontsize=8,loc="lower left"); ax.xaxis.label.set_size(8)
fig.tight_layout(); fig.savefig("fig_trap_curve.png"); plt.close(fig)
# Fig 2: scenario bars (W=4096)
df = pd.read_csv("results_b.csv"); g = df[df.window==4096].groupby(["scenario","decoder"]).ler_exact.mean().unstack()*100
sel = ["D2 Z4Z5 burst (Dhaya sim2)","R1 repo correlated (Z4 + pair45 hot)","R2 repo burst level","D3 Z3 burst (Dhaya sim3)","R3 alias-swap: Z3 hot instead of pair45","R4 alias-swap burst level"]
lab = ["Dhaya sim2\n(Z4Z5 burst)","Repo style\ncorrelated","Repo style\nburst level","Dhaya sim3\n(Z3 burst)","Alias swap\n(Z3 hot)","Alias swap\nburst level"]
fig, ax = plt.subplots(figsize=(7.2,3.7)); x=np.arange(len(sel)); w=0.26
for i,(k,c,l) in enumerate([("static prior, marginal LLR (B0)",C["B0"],"Static prior (B0)"),("MAP + AVID projection (no RL)",C["AVID"],"MAP + AVID projection"),("oracle (exact joint, true theta)",C["ORA"],"Optimal")]):
    v=[g.loc[s,k] for s in sel]; ax.bar(x+(i-1)*w,v,w,color=c,label=l)
    for xi,vi in zip(x+(i-1)*w,v): ax.text(xi,vi+0.15,f"{vi:.1f}",ha="center",fontsize=7)
ax.axvline(2.5,color="k",lw=0.8,ls=":"); ax.text(1.0,12.9,"excess mass is a pair",ha="center",fontsize=8); ax.text(4.0,12.9,"excess mass is a single qubit",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=8); ax.set_ylabel("logical error rate (%)"); ax.set_ylim(0,13.8)
ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.2)); fig.tight_layout(); fig.savefig("fig_scenarios.png"); plt.close(fig)
print("ok")
