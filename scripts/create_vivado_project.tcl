# =============================================================================
# create_vivado_project.tcl — AVID Vivado Block Design
# Project : AVID (Adaptive Variational Inference Decoder)
# Target  : Zynq-7020 (xc7z020clg484-1)
# Vivado  : 2023.1+
#
# Block design differences from ZEPHYR:
#   + weight_map_reg is a separate AXI-Lite slave (PS M_AXI_GP0 → interconnect)
#   + avid_top has TWO AXI connections: AXI-S (DMA) + AXI-Lite (weight regs)
#   + Three interrupts: DMA MM2S, DMA S2MM, epoch_irq
#
# Usage: vivado -mode batch -source scripts/create_vivado_project.tcl
# =============================================================================

set project_name "avid"
set project_dir  [file normalize "[file dirname [info script]]/../vivado_prj"]
set part         "xc7z020clg484-1"

file mkdir $project_dir
create_project $project_name $project_dir -part $part -force
set_property target_language SystemVerilog [current_project]

# --- Add RTL ---
set rtl_dir [file normalize "[file dirname [info script]]/../firmware/rtl"]
add_files -fileset sources_1 [list \
    $rtl_dir/decoder/check_node_minsum.sv \
    $rtl_dir/ingress/axis_syndrome_rx.sv  \
    $rtl_dir/ingress/axis_ingress_tap.sv  \
    $rtl_dir/decoder/axis_correction_tx.sv \
    $rtl_dir/config/weight_map_reg.sv    \
    $rtl_dir/decoder/bp_engine.sv         \
    $rtl_dir/top/aa_qec_top.sv        \
]
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1

# --- Add constraints ---
add_files -fileset constrs_1 [file normalize "[file dirname [info script]]/../firmware/constraints/avid.xdc"]

# --- Add testbench ---
set tb_dir [file normalize "[file dirname [info script]]/../firmware/tb"]
add_files -fileset sim_1 [list $tb_dir/tb_avid_top.sv]
set_property top tb_avid_top [get_filesets sim_1]

# =============================================================================
# Block Design
# =============================================================================
create_bd_design "avid_bd"

# --- Zynq PS (250 MHz PL_CLK0, S_AXI_HP0, M_AXI_GP0, 3×IRQ) ---
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {250}   \
    CONFIG.PCW_USE_S_AXI_HP0            {1}     \
    CONFIG.PCW_USE_M_AXI_GP0            {1}     \
    CONFIG.PCW_USE_FABRIC_INTERRUPT     {1}     \
    CONFIG.PCW_IRQ_F2P_INTR             {1}     \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH     {64}    \
] $ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" \
             Master "Disable" Slave "Disable"} $ps7

# --- Reset ---
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

# --- AXI DMA ---
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg               {0}  \
    CONFIG.c_m_axis_mm2s_tdata_width  {16} \
    CONFIG.c_s_axis_s2mm_tdata_width  {32} \
    CONFIG.c_mm2s_burst_size          {16} \
    CONFIG.c_s2mm_burst_size          {16} \
] $dma

# --- AXI Interconnect: PS GP0 → DMA lite + avid weight regs ---
set ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $ic

# --- Custom Decoder IP ---
set decoder [create_bd_cell -type module -reference avid_top avid_top_0]

# --- Connect clocks ---
foreach cell [list $dma $decoder $ic $rst] {
    set pins [get_bd_pins $cell/*clk* -quiet]
    if {[llength $pins]} {
        connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] $pins
    }
}

# --- Connect resets ---
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
               [get_bd_pins proc_sys_reset_0/ext_reset_in]
set rst_n [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net $rst_n [get_bd_pins axi_dma_0/axi_resetn]
connect_bd_net $rst_n [get_bd_pins avid_top_0/aresetn]

# --- GP0 → Interconnect → DMA lite + Decoder AXI-Lite ---
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
                    [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] \
                    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
# M01_AXI → avid_top AXI-Lite (weight registers)
# Manual port connection required after packaging avid_top as custom IP

# --- AXI-Stream: DMA MM2S → Decoder syndrome ---
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
                    [get_bd_intf_pins avid_top_0/s_axis_syndrome]

# --- AXI-Stream: Decoder correction → DMA S2MM ---
connect_bd_intf_net [get_bd_intf_pins avid_top_0/m_axis_correction] \
                    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# --- DMA → PS HP0 (high-bandwidth DDR path — no CPU) ---
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
                    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
                    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# --- Interrupts: DMA IRQ + epoch_irq → PS IRQ_F2P ---
# Concat DMA mm2s_introut, s2mm_introut, epoch_irq → IRQ_F2P[2:0]
set xlconcat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0]
set_property -dict [list CONFIG.NUM_PORTS {3}] $xlconcat
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut]  [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins axi_dma_0/s2mm_introut]  [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins avid_top_0/epoch_irq]  [get_bd_pins xlconcat_0/In2]
connect_bd_net [get_bd_pins xlconcat_0/dout]         [get_bd_pins processing_system7_0/IRQ_F2P]

# =============================================================================
# Address assignment
# =============================================================================
assign_bd_address [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg]
set_property offset 0x40000000 [get_bd_addr_segs \
    {processing_system7_0/Data/SEG_axi_dma_0_Reg}]
set_property range 64K [get_bd_addr_segs \
    {processing_system7_0/Data/SEG_axi_dma_0_Reg}]

assign_bd_address [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM]

# =============================================================================
# Validate and export
# =============================================================================
validate_bd_design
save_bd_design
make_wrapper -files [get_files avid_bd.bd] -top
add_files -norecurse $project_dir/${project_name}.srcs/sources_1/bd/avid_bd/hdl/avid_bd_wrapper.v
set_property top avid_bd_wrapper [current_fileset]

set_property strategy "Performance_ExplorePostRoutePhysOpt" [get_runs impl_1]

puts "\n=== AVID Vivado Project Created ==="
puts "  Part    : $part"
puts "  Clock   : 250 MHz"
puts "  Decode  : 17 cycles = 68 ns"
puts "  Weights : AXI-Lite @ 0x43C01000 (to be assigned)"
puts "  Next steps:"
puts "    launch_runs synth_1 -jobs 8"
puts "    launch_runs impl_1  -jobs 8"
puts "    open_run impl_1; report_timing_summary"
