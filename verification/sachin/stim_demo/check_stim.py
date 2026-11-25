import numpy as np
from scipy import stats
from avid_model import *
from avid_stim import *
th1=theta_uniform(0.01); th2=th1.copy(); th2[idx_edge(4,5)]=0.08; th3=th1.copy(); th3[idx_single(3)]=0.08
rep=theta_prior(); rep[idx_single(4)]=0.055; rep[idx_edge(4,5)]=0.075
cases={"Dhaya sim1":th1,"Dhaya sim2 (Z4Z5 burst)":th2,"Dhaya sim3 (Z3 burst)":th3,"repo-style correlated":rep}
N=4_000_000
for name,th in cases.items():
    s,l=sample_syn_ell(th,N,seed=123)
    H=hist_joint(s,l); P=dp_joint(th)
    E=P*N; mask=E>5
    chi2=((H[mask]-E[mask])**2/E[mask]).sum(); dof=mask.sum()-1
    z=(H-E)/np.sqrt(np.maximum(E,1e-9)*(1-P))
    emp_opt=np.minimum(H[:,0],H[:,1]).sum()/N
    print(f"{name:26s} chi2={chi2:6.1f} dof={dof} p={1-stats.chi2.cdf(chi2,dof):.3f} max|z|={np.abs(z[mask]).max():.2f} | LER_opt analytic={ler_opt(P):.5f} stim-empirical={emp_opt:.5f}")
