# RL Link Adaptation (DQN)

This folder contains a drop‑in reinforcement learning policy to replace ILLA/OLLA, traditional link adaptation algorithms. Uses Sionna library.

## Files
- `rl_link_adaptation_sionna.ipynb` — main Jupyter notebook to run training.
- `agent_dqn.py` — minimal DQN agent (TensorFlow/Keras) with replay buffer.
- `la_env.py` — thin adapter around the Sionna tutorial that exposes two callables:
  - `get_effective_sinr_db()` → float (current effective SINR in dB)
  - `send_with_mcs(mcs_idx)` → `(ack:int, sinr_eff_db:float)`
- `mcs_table.py` — NR MCS spectral efficiency helper.
- `nr_link_full` - creates realistic channel

## How to use
1. Open this folder in JupyterLab.
2. Open `rl_link_adaptation_sionna.ipynb` and run the cells top to bottom.
3. Set agent exploration parameter to 0.1 for training, to 0.0 for using agent.
4. The agent will replace ILLA/OLLA and learn from ACK/NACK feedback.


