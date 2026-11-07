# AVID Repository Tools & Verification Utilities (`tools/`)

- `validate_architecture.py`: Validates machine-readable YAML contracts, CSS parity-check commutativity, and syndrome signatures.
- `validate_repository.py`: Validates canonical repository structure, absence of duplicate documentation, and absence of local `file://` links.
- `verify_v0_closure_40.py`: Executes all 40 V0 mathematical, numerical, and architectural closure checks (`A01..D10`).
- `verify_provenance.py`: Verifies that all reports in `reports/` carry explicit simulator and CPU emulator provenance labels.
- `verify_requirements.py`: Verifies `architecture/verification/requirements_traceability.yaml`.
- `run_experiment.py`: Python wrapper around `scripts/run_all.sh` for running a YAML experiment configuration.
- `compare_reports.py`: Compares two JSON execution reports across M1, M3, and M4 scenarios.
- `generate_summary.py`: Generates a concise Markdown/JSON summary from `reports/latest_report.json`.
- `collect_artifacts.py`: Validates experiment YAML schema and packages run evidence into `runs/<run-id>/`.
- `generate_code_artifacts.py`: Generates `architecture/code_model/` artifacts (`Hx.json`, `Hz.json`, `tanner_tables.json`).
- `collect_target_timing.py`: Profiles slow-loop L-BFGS timing.
- `cloudshell_hf_pipeline.py`: V0 world-model and LoRA offline training pipeline.
