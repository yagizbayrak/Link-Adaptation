# Single clock constraint for the out of context timing run, 400 MHz on the Kria K26 module.
create_clock -period 2.500 -name clock [get_ports clock]

set_false_path -from [get_ports reset]
