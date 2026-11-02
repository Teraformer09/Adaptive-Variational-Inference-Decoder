# 08 — Weighted-Bellman R2D2 (`R2D2-Graph-RSSM`) and Spatio-Temporal World Model Mathematics

## 1. Non-Vision Bipartite Stabilizer-Data Graph Perception Layer
At each slow-loop observation window $\tau \in \{1, \dots, T\}$, the raw syndrome stream is summarized into node feature tokens $x_\tau \in \mathbb{R}^{V \times d_{\text{in}}}$ on the bipartite quantum code graph $\mathcal{G} = (\mathcal{V}_{\text{data}} \cup \mathcal{V}_{\text{check}}, \mathcal{E}_{\text{Tanner}} \cup \mathcal{E}_{ZZ})$. Multi-head graph attention masked by the topological adjacency matrix $A \in \{0, 1\}^{V \times V}$ computes structurally grounded node embeddings:
$$\alpha_{ij}^{(h)} = \frac{A_{ij} \exp\!\left(\frac{(W_Q^{(h)} h_i)^\top (W_K^{(h)} h_j)}{\sqrt{d_k}}\right)}{\sum_{u : A_{iu}=1} \exp\!\left(\frac{(W_Q^{(h)} h_i)^\top (W_K^{(h)} h_u)}{\sqrt{d_k}}\right)}, \qquad z_\tau = \operatorname{Pool}_{i \in \mathcal{V}}\!\left(\Big\|_{h=1}^H \sum_{j=1}^V \alpha_{ij}^{(h)} W_V^{(h)} h_j\right) \in \mathbb{R}^{d_{\text{model}}}$$

---

## 2. Recurrent State-Space Dynamics (`R2D2-Graph-RSSM`) with Burn-In Warmup
Following the Recurrent Experience Replay in Distributed Reinforcement Learning (**R2D2**) architecture, each replay trajectory of length $T_{\text{total}} = T_{\text{burn}} + T_{\text{unroll}}$ is split into:
1. **Burn-In Phase ($\tau = 1, \dots, T_{\text{burn}}$)**: Unrolls the deterministic GRU recurrent state $h_\tau$ from stored recurrent state $h_0$ with `stop_gradient` to eliminate recurrent state staleness:
   $$h_\tau = \operatorname{GRU}_\phi(h_{\tau-1}, [z_\tau \| s_{\tau-1}]), \qquad s_\tau \sim q_\phi(s_\tau \mid h_\tau, z_\tau) = \mathcal{N}\!\left(\mu_\phi^{\text{post}}(h_\tau, z_\tau), \, \operatorname{diag}(\sigma_\phi^{\text{post}}(h_\tau, z_\tau)^2)\right)$$
2. **Active Learning Unroll ($\tau = T_{\text{burn}}+1, \dots, T_{\text{total}}$)**: Evaluates the transition prior $p_\psi(s_\tau \mid h_\tau) = \mathcal{N}(\mu_\psi^{\text{prior}}(h_\tau), \operatorname{diag}(\sigma_\psi^{\text{prior}}(h_\tau)^2))$ and the variational Evidence Lower Bound (ELBO) KL divergence:
   $$\mathcal{L}_{\text{KL}}(\tau) = D_{\text{KL}}\!\left(q_\phi(s_\tau \mid h_\tau, z_\tau) \,\|\, p_\psi(s_\tau \mid h_\tau)\right)$$

---

## 3. Uncertainty-Weighted $n$-Step Bellman Return Objective (Weighted-Bellman R2D2)
Let $r_\tau \in \mathbb{R}$ denote the instantaneous quantum decoding survival reward at window $\tau$:
$$r_\tau = 1 - 2\,\widehat{\text{LER}}_\tau(\hat{\Gamma}_\tau) - \eta_{\text{NLL}} \mathcal{L}_{\text{Joint-DP}}(\hat{\theta}_\tau)$$
To handle large variations in reward magnitude across stationary and burst regimes without clipping gradients, we apply the Kapturowski invertible value rescaling function $h: \mathbb{R} \to \mathbb{R}$ and its exact analytical inverse $h^{-1}$:
$$h(u) = \operatorname{sign}(u)\!\left(\sqrt{|u| + 1} - 1\right) + \epsilon u, \qquad h^{-1}(v) = \operatorname{sign}(v)\!\left[ \left(\frac{\sqrt{1 + 4\epsilon(|v| + 1 + \epsilon)} - 1}{2\epsilon}\right)^2 - 1 \right] \quad (\epsilon = 10^{-2})$$

The $n$-step ($n = 3$) bootstrapped Bellman target $y_\tau^{(n)}$ using target value network $V_{\bar{\psi}}$ is:
$$G_\tau^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{\tau+k} + \gamma^n h^{-1}\!\left(V_{\bar{\psi}}(h_{\tau+n}, s_{\tau+n})\right), \qquad y_\tau^{(n)} = h\!\left(G_\tau^{(n)}\right)$$

Let $\delta_\tau = V_\psi(h_\tau, s_\tau) - y_\tau^{(n)}$ be the temporal-difference (TD) Bellman error and $\hat{\sigma}_{\text{unc}}^2(h_\tau, s_\tau)$ be the epistemic/aleatoric variance predicted by the uncertainty head. The **Weighted-Bellman R2D2 Loss** weights each Bellman backup by the combined R2D2 priority $|\delta_\tau|^p$ ($p = 0.9$) and inverse uncertainty precision $\hat{\sigma}_{\text{unc}}^{-2}(h_\tau, s_\tau)$:
$$w_\tau = \frac{\operatorname{sg}\!\left(|\delta_\tau|^p + \hat{\sigma}_{\text{unc}}^{-2}(h_\tau, s_\tau)\right)}{\frac{1}{T_{\text{unroll}}} \sum_{\tau'} \operatorname{sg}\!\left(|\delta_{\tau'}|^p + \hat{\sigma}_{\text{unc}}^{-2}(h_{\tau'}, s_{\tau'})\right)}, \qquad \mathcal{L}_{\text{WB-R2D2}} = \frac{1}{T_{\text{unroll}}} \sum_{\tau=T_{\text{burn}}+1}^{T_{\text{total}}} w_\tau \cdot \left(V_\psi(h_\tau, s_\tau) - y_\tau^{(n)}\right)^2$$

---

## 4. Pair-Aware Logical-Triangle LLR Projection & Regime LoRA Adaptation
The parameter prediction head outputs the 21-dimensional physical noise estimate $\hat{\theta}_\tau = \sigma(W_\theta [h_\tau \| s_\tau] + b_\theta) \in (0, 0.45)^{21}$ and direct Q7.8 residual LLR corrections $\Delta \hat{\Gamma}_\tau \in \mathbb{R}^9$. For every nearest-neighbor edge $e = (u, v) \in \mathcal{E}_{ZZ}$ forming a weight-3 logical triangle $\{u, v, k\}$, the joint correlated pair probability $p_{uv}^{\text{joint}} = \hat{p}_{uv}(1 - \hat{p}_u)(1 - \hat{p}_v) + \hat{p}_u \hat{p}_v(1 - \hat{p}_{uv})$ is projected into the variable-node LLRs as:
$$\gamma_u \leftarrow \min\!\left(\gamma_u, \; 0.48 \ln\frac{1 - p_{uv}^{\text{joint}}}{p_{uv}^{\text{joint}}}\right), \qquad \gamma_v \leftarrow \min\!\left(\gamma_v, \; 0.48 \ln\frac{1 - p_{uv}^{\text{joint}}}{p_{uv}^{\text{joint}}}\right)$$
ensuring $\Gamma_u + \Gamma_v < \Gamma_k$ whenever the correlated pair fault $(u, v)$ is more probable than its single-qubit alias $k$.

Across the 5 non-stationary hardware noise regimes $r \in \{1, \dots, 5\}$, fast regime specialization is achieved via rank-$k_{\text{LoRA}}$ ($k_{\text{LoRA}} = 4$) Low-Rank Adaptation matrices $B_r \in \mathbb{R}^{d_{\text{out}} \times 4}, A_r \in \mathbb{R}^{4 \times d_{\text{in}}}$:
$$W_r = W_0 + \frac{\alpha_{\text{LoRA}}}{k_{\text{LoRA}}} B_r A_r$$

---

## References
- **[R4]** E. van den Berg et al., *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R6]** J. Chen and M. P. C. Fossorier, *IEEE Transactions on Communications*, vol. 50, no. 3, pp. 406–414, 2002.
- **[R10]** S. Kapturowski, G. Ostrovski, J. Quan, R. Munos, and W. Dabney, "Recurrent experience replay in distributed reinforcement learning (R2D2)," *International Conference on Learning Representations (ICLR)*, 2019.
- **[R11]** D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, "Dream to control: Learning behaviors by latent imagination (RSSM)," *ICLR*, 2020.
- **[R12]** E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," *ICLR*, 2022.
