# Independent Mathematical Oracles (`tests/layer3/set5_control_mathematics/oracles/`)

This directory owns strictly independent Python reference oracles for Set 5 (`L3-MATH-001..062`):

1. `brute_force_likelihood.py` — Independent $2^{21}$ / exact subset state-space enumeration of $P(s \mid \theta)$ and empirical cross-entropy NLL without importing or wrapping `software/src/inference/noise_model.cpp`.
2. `finite_difference_gradient.py` — Independent central finite-difference $\epsilon$-sweep gradient checker against analytical coset-inversion gradients.
3. `identifiability_reference.py` — Independent Jacobian $J_{ij} = \partial P(s_i \mid \theta) / \partial \theta_j$, SVD rank, null-space basis (`0x0` zero-signature edges `ZZ_3_6, ZZ_2_5` in X-basis and `XX_0_1, XX_7_8` in Z-basis), and 9-orbit quotient map.
4. `q78_reference.py` — Independent IEEE-754 / exact rational reference for $\operatorname{round}(2^8 \log((1-p)/p))$ and saturation/sign rules.

Per the Layer 3 Execution Directive, executable Set 5 tests and oracle runs remain gated until Set 1 (`G1`) review completes.
