# Exact Joint Correlated Likelihood Specification (`A02 / Q03`)

## 1. Objective Function
Given an empirical burst of $N$ syndrome packets with 16-bin histogram counts $c(s)$ and empirical frequencies $\hat{P}(s) = c(s)/N$ over $s \in \{0, \dots, 15\}$, the estimator minimizes the prior-tethered exact cross-entropy objective:
$$\mathcal{L}(\theta) = -\sum_{s=0}^{15} \hat{P}(s) \log P_\theta(s) + \frac{\lambda}{2} \|\theta - \theta_{\text{prior}}\|_2^2$$
where $P_\theta(s)$ is computed by the 16-state XOR-convolution dynamic program over all 21 physical fault mechanisms.

## 2. Complexity & Real-Time Budget on ARM Cortex-A53
- **Forward DP pass**: $21 \times 16 = 336$ floating-point multiply-adds ($< 0.4\,\mu\text{s}$).
- **Exact analytic gradient pass**: $21 \times 16 = 336$ floating-point operations via $\frac{\partial P(s)}{\partial p_e} = \frac{P(s \oplus g_e) - P(s)}{1 - 2 p_e}$ ($< 0.4\,\mu\text{s}$).
- **10-iteration L-BFGS update**: $< 15\,\mu\text{s}$ total on Cortex-A53 @ 1.2 GHz, well inside the slow-loop re-estimation budget.

## References
- **[R4]** E. van den Berg et al., *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R7]** D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," *Mathematical Programming*, vol. 45, pp. 503–528, 1989.
