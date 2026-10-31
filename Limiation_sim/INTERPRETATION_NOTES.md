# Interpretation Notes for the Existing QuTiP Limitation Experiments

This note accompanies the existing scripts without modifying or deleting their outputs.

## Test 1 — coherent ZZ and crosstalk

The current script computes projections of an evolved two-qubit state onto `|-->` and onto single-qubit `|->` subspaces. Those are valid state-projection statistics for the specified preparation, Hamiltonian and measurement basis. They are **not automatically Pauli-Z fault probabilities**. In coherent evolution with ZZ coupling plus spectator drive, a projection probability cannot be identified with an effective Pauli fault without a defined gate/circuit twirl or channel-tomography reduction. Rename these outputs as projection probabilities unless such a reduction is added.

The correlated Lindblad operator proportional to `Z1 Z2` is not channel-equivalent to coherent ZZ evolution. Matching one single-cycle scalar probability does not establish equivalence. A defensible comparison should use process/Pauli-transfer matrices or a defined set of QEC-relevant input states and observable trajectories.

The hard-coded `92 kHz` ZZ and `25 kHz` spectator-drive terms are exploratory model settings in this script; they are not established here as measured Rigetti device parameters.

## Test 2 — leakage and scaling

The 8 ns Gaussian qutrit pulse result is specific to the chosen pulse envelope, anharmonicity and rotating-frame Hamiltonian. It is an illustrative leakage calculation, not a universal per-gate leakage rate. The leakage reproduction benchmark in Varbanov et al. used `quantumsim`; the present QuTiP model is not automatically a reproduction of that paper.

`3^(2N) * 16 bytes` is the storage requirement for an explicitly dense complex128 density matrix. It is not a universal memory requirement for every QuTiP solver path. The current `tracemalloc` measurements may not include all native allocations from numerical libraries; use process RSS/peak-RSS measurements as a second measurement.

## Test 3 — non-stationarity and memory

A changing classical OU/TLS trajectory is useful for testing drift estimation, but the current per-round simulation resets the qubit state for each round. It therefore does not represent one continuously evolving qubit–environment trajectory across all rounds. The Ramsey ensemble samples detuning at trajectory start and holds it during each Ramsey evolution; this is quasi-static ensemble dephasing, not continuous OU/TLS switching during the Ramsey interval.

The timing comparison between one `mesolve` call and many separate calls measures the overhead of those particular call patterns. It does not prove that QuTiP is intrinsically unable to represent non-stationary noise. Keep the result scoped to the tested implementation.

## Plot provenance

Any IQ clouds, flux tails, quasiparticle-burst profiles, decoder-latency curves, throughput values or FPGA latency claims that are manually generated or hard-coded must be labelled `illustrative` or `synthetic`, unless the repository includes traceable device measurements, scripts and calibration metadata supporting them. In particular, do not present the `68 ns` FPGA value or derived fidelity as measured hardware performance without measurement provenance.
