# AVID Model Checkpoints & Quantized Weight Banks

- `r2d2_v1_trained.bin`, `r2d2_v1_trained.json`, `r2d2_v1_trained.onnx`: Canonical V1 Python-trained 64-unit Dueling LSTM R2D2 controller checkpoint (`19,744` parameters = `19.28 KB` `int8` / `77.12 KB` `fp32`, `157,984 B` FP64 binary with SHA-256 verification and `< 1e-12` Python-to-C++ parity).
- `avid_joint_dp_checkpoints.json`: Pre-computed 21-parameter Joint-DP MAP reference solutions across regimes `R1–R5`.
- `baseline_vs_avid_benchmark.json`: V0 250,000-round (`50,000` rounds/regime) shared-stream model benchmark (`B0`, `B1`, `AVID Joint-DP`, `AVID + R2D2-WB`, `B4 Oracle`).
- `q78_weight_banks.npz`: Signed Q7.8 hardware weight banks for initial boot and fallback recovery.
- `weighted_bellman_r2d2_backbone.pt`: Trained Weighted-Bellman R2D2 recurrent world-model checkpoint (V0 offline training scope).
- `spatiotemporal_transformer_backbone.pt` & `full_world_model_bundle.pt`: V0 offline model exploration checkpoints (`BM2` ablation scope).
- `lora_adapters/R1..R5.pt`: Rank-4 regime-specific LoRA adapter checkpoints evaluated in `Benchmark-Eval/Baseline_V0/BM2_Architectural_and_Loss_Ablation/`.
