Repo: Teraformer09/Adaptive-Variational-Inference-Decoder, commit 54da68d92778e4e1ffde9514aaaae585638761ff (origin/AVID_EPJ_Patches)

1) One-round test case (trajectory test_traj_00_s30000, round 208):
   cd <repo> && git checkout 54da68d
   python3 -I testcase/one_round_case.py <repo> 208
   Full-trajectory scoring: see testcase/trajectory_scoring.log (command embedded in the log header, CausalStreamRunner, window 512, commit delay 20, K=3)

2) Stim/PyMatching demo (scripts are outside the repo):
   pip install stim pymatching numpy scipy pandas matplotlib
   cd stim_demo && python3 check_dhaya.py && python3 check_stim.py && python3 check_decoders.py && python3 run_b.py && python3 run_trap.py && python3 make_figs.py
   Hot pair = scenario R1 (q4=0.055, pair 4-5=0.075), R2 burst level. Hot single = R3 (q4=0.055, q3=0.075), R4.
   Seeds: Stim sampler 1000+scenario index, 40 estimation seeds, trap curve seeds 77 / 5 / 99.
   Daemon scenarios: correlated_noise vs alias_single_hot (daemon_scenario_patch.diff), seeds 2026-2035, window 256, 4096 rounds.
