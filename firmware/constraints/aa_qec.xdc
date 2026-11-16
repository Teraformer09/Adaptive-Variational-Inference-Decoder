# =============================================================================
# Constraints : aa_qec.xdc
# Project     : AVID (Adaptive Variational Inference Decoder)
# Target      : Xilinx Zynq-7020 (xc7z020clg484-1) / ZCU104 / Kria K26
# Target clock: 250 MHz (4.000 ns period) — COMMANDMENT #2
# =============================================================================
# Primary 250 MHz fabric clock constraint for standalone aa_qec_top synthesis/STA.
# (When integrated inside a Zynq PS block design, PL_CLK0 supplies this 4.000 ns clock.)
create_clock -period 4.000 -name pl_clk0 [get_ports aclk]

# --- False paths: AXI-Lite config registers are quasi-static (Slow-Path) ---
# The ARM writes LLRs at ~10ms intervals; the commit-synchronized swap path
# gates bank updates synchronously via bp_ready_in == 1 in the single 250 MHz domain.

# --- Pblock: keep all check_node_minsum PEs adjacent for short routing ---
# (Uncomment after initial placement to lock down the systolic array)
# create_pblock pblock_systolic
# add_cells_to_pblock [get_pblocks pblock_systolic] \
#     [get_cells -hier -filter {NAME =~ *gen_check_nodes*}]
# resize_pblock pblock_systolic -add {SLICE_X0Y80:SLICE_X25Y105}

# --- ILA debug probes (uncomment to insert ILA in Vivado) ---
# set_property MARK_DEBUG true [get_nets {u_weights/swap_pending}]
# set_property MARK_DEBUG true [get_nets {u_weights/epoch_count[*]}]
# set_property MARK_DEBUG true [get_nets {u_bp_x/state[*]}]
# set_property MARK_DEBUG true [get_nets {m_axis_correction_tvalid}]

# --- Implementation strategy for 250 MHz closure ---
# Set in Vivado: Implementation -> Strategy -> Performance_ExplorePostRoutePhysOpt
# set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
# set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
