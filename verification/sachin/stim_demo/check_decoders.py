import numpy as np
from avid_model import *; from avid_decoders import *
rng=np.random.default_rng(0)
# 1. analytic likelihood/gradient vs DP and finite differences
th=rng.uniform(0.005,0.1,M); P=p_syndrome(th); print("WHT P(s) vs DP marginal: max diff %.2e"%np.abs(P-dp_joint(th).sum(1)).max())
phat=rng.dirichlet(np.ones(16)); th0=theta_prior()
f,g=nll_and_grad(th,phat,th0,0.05); fd=np.array([(nll_and_grad(th+e*1e-6,phat,th0,.05)[0]-nll_and_grad(th-e*1e-6,phat,th0,.05)[0])/2e-6 for e in np.eye(M)])
print("analytic grad vs central FD: max diff %.2e"%np.abs(g-fd).max())
# 2. identifiability: rank of Jacobian of P(s) wrt theta at an interior point (expect 9 non-trivial orbits)
J=np.array([( p_syndrome(th+e*1e-6)-p_syndrome(th-e*1e-6))/2e-6 for e in np.eye(M)]).T
print("rank of dP/dtheta (16x21):", np.linalg.matrix_rank(J,tol=1e-8), "(expect 9 orbits, hist sums to 1 -> 9 free dims)")
# 3. PyMatching vs my min-weight decoder on same (float) marginal weights, many random thetas
mism=0; tot=0
for _ in range(300):
    th=rng.uniform(0.003,0.15,M); a=matching_table(th); b=minweight_table(np.log((1-effective_p(th))/effective_p(th)))
    mism+=int((a!=b).sum()); tot+=16
print("PyMatching vs min-weight table: %d / %d syndrome decisions differ"%(mism,tot))
# 4. MAP fit recovers theta when data is plentiful and orbit-observable
true=theta_prior(); true[idx_single(4)]=0.055; true[idx_edge(4,5)]=0.075
Pt=p_syndrome(true); fit=fit_map(Pt*1.0)  # infinite-data limit
print("infinite-data fit: max |P(s) fit - truth| = %.2e ; max |theta_fit - theta_true| = %.3f (nonidentifiable dirs -> prior)"%(np.abs(p_syndrome(fit)-Pt).max(), np.abs(fit-true).max()))
print("   theta_fit at hot params: Z4=%.3f pair45=%.3f Z3=%.3f Z0=%.3f"%(fit[4],fit[idx_edge(4,5)],fit[3],fit[0]))
