# AVID Python Simulation & Reference Models (`simulation/`)

- `reference/code_model/css_code_builder.py`: Rotated $d=3$ and $d=5$ CSS surface code matrices and Tanner tables.
- `reference/noise_model/`: `joint_dp_likelihood.py` (16-state XOR-convolution DP) and `identifiability_svd.py` (Jacobian SVD & gauge orbit analysis).
- `reference/baselines/`: Fair shared-input baselines `b0_static_bp.py` through `b4_oracle_bp.py`.
- `stimulus/`: `noise_generator.py`, `packet_codec.py`, and `stim_to_bin.py`.
- `tests/`: Pytest suite verifying geometry, signatures, identifiability, Joint-DP likelihood, and packet round-trip.
