# Central Operational Logs (`logs/`)

This directory stores operational logs for environment setup, builds, repository validation, CI checks, and maintenance scripts.
**Important Policy**: Immutable per-experiment execution evidence (Verilator waveforms, QEMU scenario logs, bridge JSONL traces, and run reports) belongs exclusively in `runs/<run-id>/`, never in `logs/`.
