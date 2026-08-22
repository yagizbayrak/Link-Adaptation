PYTHON = /home/yagiz/miniforge3/envs/linkadapt/bin/python
IVERILOG = /home/yagiz/miniforge3/envs/hdl/bin/iverilog
VVP = /home/yagiz/miniforge3/envs/hdl/bin/vvp

trace:
	$(PYTHON) scripts/trace_scene.py --scene san_francisco
	$(PYTHON) scripts/trace_scene.py --scene munich
	$(PYTHON) scripts/trace_scene.py --scene street_canyon

calibrate:
	$(PYTHON) scripts/calibrate_report.py --scene san_francisco

train:
	$(PYTHON) scripts/train.py --scene san_francisco --episodes 10

evaluate:
	$(PYTHON) scripts/evaluate.py --scene san_francisco
	$(PYTHON) scripts/evaluate.py --scene munich

figures:
	$(PYTHON) scripts/make_figures.py

test:
	$(PYTHON) -m pytest tests/ -q

export:
	$(PYTHON) scripts/export_rtl.py

rtl:
	$(IVERILOG) -g2012 -o build/tb_illa rtl/illa.sv tb/tb_illa.sv
	$(VVP) build/tb_illa
	$(IVERILOG) -g2012 -o build/tb_olla rtl/illa.sv rtl/olla.sv tb/tb_olla.sv
	$(VVP) build/tb_olla
	$(IVERILOG) -g2012 -o build/tb_dqn rtl/dqn.sv tb/tb_dqn.sv
	$(VVP) build/tb_dqn
	$(IVERILOG) -g2012 -o build/tb_dqn_la rtl/illa.sv rtl/dqn.sv rtl/dqn_la.sv tb/tb_dqn_la.sv
	$(VVP) build/tb_dqn_la

synth:
	vivado -mode batch -source scripts/synth.tcl -nojournal -log build/vivado.log

all: trace calibrate train evaluate figures

.PHONY: trace calibrate train evaluate figures test export rtl synth all
