# AVID independent Stim cross-check (d=3, Z errors, one round)

Question: does AVID's gain survive noise that the team's own generator did not produce, and where does it come from?

Setup: pip install stim pymatching numpy scipy pandas matplotlib

Run in this order (about 1 minute total):
  python3 check_dhaya.py      # reproduces Dhaya's table; DP vs 2^21 brute force
  python3 check_stim.py       # Stim P(s,l) vs analytic (chi-square)
  python3 check_decoders.py   # gradient, rank 9, PyMatching vs min-weight table
  python3 run_b.py            # 8 scenarios x 4 windows x 40 seeds -> results_b.csv
  python3 run_trap.py         # same-histogram "alias trap" curve -> trap_curve.csv, Stim failure-count check
  python3 make_figs.py        # fig_trap_curve.png, fig_scenarios.png
Then compile avid_stim_crosscheck.tex (needs the two PNGs).

Files
  avid_model.py     code, 21 mechanisms (repo C++ check layout, asserted against NM_EVENT_SIGNATURES), exact DP
  avid_stim.py      Stim circuit (independent sampler)
  avid_decoders.py  16-state MAP estimator, Python port of to_pair_projected_q7_8_llr, min-weight / joint / PyMatching tables

Caveats: one-round code-capacity noise; the projection is a Python port (from branch AVID_EPJ_Patches, commit 54da68d);
decoding is exact minimum-LLR over 512 patterns, not the RTL BP+OSD-0; the RL controller is not tested.
