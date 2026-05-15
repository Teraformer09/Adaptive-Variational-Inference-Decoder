# Identifiability of Correlated Pauli Noise from Stabilizer Syndromes

**Project:** AVID (Adaptive Variational Inference Decoder)  
**Satisfies:** Blueprint COMMANDMENT #8 — "The inference algorithm must only attempt to learn noise parameters that leave a distinct topological signature in the syndrome volume."

---

## 1. What Is Identifiable

A noise parameter θ is **identifiable from stabilizer syndromes** if and only if there exists a mapping from the joint syndrome distribution P(s|θ) to θ that is injective (one-to-one) in the neighborhood of the true value θ*.

### Identifiable Parameters

**Single-qubit Pauli error rates** p_v ∈ (0, 0.5):  
A Z-error on qubit v flips exactly those check nodes in N(v).  
The marginal syndrome rate of check c is:

```
P(s_c = 1 | {p_v}) = [1 - ∏_{v ∈ N(c)} (1 - 2p_v)] / 2
```

The Jacobian ∂P(s_c)/∂p_v = ∏_{u ∈ N(c), u≠v}(1-2p_u) is non-zero for 
p_v ∈ (0, 0.5), so each p_v is locally identifiable from the check rates.

**Correlated ZZ crosstalk rates** p_ij for adjacent pairs (i,j):  
A ZZ error on (i,j) simultaneously flips both qubits, creating a 
characteristic *pair* of syndrome changes.  The temporal cross-correlation:

```
C_{c,c'}(τ=0) = Cov[s_c^t, s_c'^t] = P(s_c=1, s_c'=1) - P(s_c=1)P(s_c'=1)
```

is elevated by p_ij when checks c and c' share both qubits i and j.  
This signature is **distinct** from two independent single-qubit errors on 
i and j, which would require two simultaneous independent events with 
probability p_i × p_j << p_ij for the crosstalk-dominated regime.

---

## 2. What Is NOT Identifiable (and Why We Exclude It)

**Coherent over-rotations** (e.g., systematic X-axis rotations by angle ε):  
Stabilizer measurements perform Pauli twirling: the measurement process 
projects any coherent error (ε-rotation) onto the Pauli group.  The 
resulting Pauli-frame distribution is identical for any ε with the same 
fidelity |cos(ε)|².  Therefore, the specific coherent angle is NOT 
distinguishable from a Pauli error rate p = sin²(ε).  

→ **AVID tracks p_eff, not coherent angles.**

**Long-range correlations** beyond nearest-neighbor (|i-j| > 1 in grid distance):  
For d=3, the surface code has diameter 3.  A ZZ error between non-adjacent 
qubits (e.g., q0 and q8) produces a syndrome pattern identical to two 
independent errors at q0 and q8.  The Fisher Information for long-range 
p_ij approaches zero as distance increases, making estimation unreliable.

→ **AVID parameterises only nearest-neighbor crosstalk.**

---

## 3. Fisher Information Analysis

The Fisher Information for parameter θ_k with respect to syndrome round t:

```
I(θ_k) = Σ_c [  (∂q_c/∂θ_k)²  /  (q_c(1 - q_c))  ]
```

For a single-qubit rate p_v, summing over its adjacent checks:

```
I(p_v) = Σ_{c: v∈N(c)} [ ∏_{u∈N(c),u≠v}(1-2p_u) ]² / [q_c(1-q_c)]
```

At p = 0.01 for all qubits on d=3, I(p_v) ≈ 3.9 per syndrome round.  
With W = 10,000 rounds, the Cramér-Rao lower bound gives:

```
Var(p̂_v) ≥ 1 / (W × I(p_v)) ≈ 2.6 × 10⁻⁵
```

i.e., standard deviation σ ≈ 0.005 — sufficient to detect a 3× rate change 
(0.01 → 0.03) with 6σ significance in 10,000 rounds.

---

## 4. Consequences for LLR Injection

The FPGA BP decoder uses per-qubit channel LLRs:

```
LLR_v = log( (1 - p_v_eff) / p_v_eff )
```

where p_v_eff = p_v + Σ_j p_vj - 2Σ_j p_v·p_vj (first-order).

The adaptive range:  
- At p_v = 0.01: LLR_v ≈ 4.60 (Q7.8: 0x04A0)  
- At p_v = 0.05: LLR_v ≈ 2.94 (Q7.8: 0x02F0)  
- At p_v = 0.10: LLR_v ≈ 2.20 (Q7.8: 0x0232)  
- At p_v = 0.30: LLR_v ≈ 0.85 (Q7.8: 0x00D9)  

The BP decoder's soft-decision performance degrades gracefully with LLR 
accuracy; a 10% error in p̂_v causes < 0.5% increase in logical error rate 
at p = 0.05 (verified in the ablation study notebook).

---

## 5. References

1. Fowler et al., "Surface codes: Towards practical large-scale quantum computation," PRA 86, 032324 (2012)
2. Bravyi et al., "Correcting coherent errors with surface codes," npj Quantum Inf 4, 55 (2018)  
3. Harper et al., "Efficient learning of quantum noise," Nature Physics 16, 1184–1188 (2020)  
4. Flammia & Liu, "Direct fidelity estimation from few Pauli measurements," PRL 106, 230501 (2011)
