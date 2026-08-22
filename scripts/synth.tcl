# Out of context synthesis and implementation of the learned link adaptation core on the Kria K26 module, reporting timing and utilisation.

set part xck26-sfvc784-2LV-c
set top dqn_la
set outdir build/vivado

file mkdir $outdir

read_verilog -sv rtl/illa.sv rtl/dqn.sv rtl/dqn_la.sv
read_xdc constraints/dqn_la.xdc

synth_design -top $top -part $part -mode out_of_context -directive PerformanceOptimized -retiming

opt_design
place_design -directive ExtraTimingOpt
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore

report_utilization -file $outdir/utilisation.rpt
report_timing_summary -delay_type min_max -max_paths 10 -file $outdir/timing.rpt
report_clock_utilization -file $outdir/clocks.rpt

set slack [get_property SLACK [get_timing_paths -delay_type max]]
puts "worst negative slack: $slack ns"

if {$slack < 0} {
    puts "TIMING NOT MET at 2.500 ns"
} else {
    puts "TIMING MET at 2.500 ns"
}

write_checkpoint -force $outdir/$top.dcp
