# Sionna RL Link Adaptation (DQN)

This folder contains a drop‑in reinforcement learning policy to replace ILLA/OLLA
in the Sionna Link Adaptation tutorial.

## Files
- `rl_link_adaptation.ipynb` — main Jupyter notebook to run training.
- `agent_dqn.py` — minimal DQN agent (TensorFlow/Keras) with replay buffer.
- `la_env.py` — thin adapter around the Sionna tutorial that exposes two callables:
  - `get_effective_sinr_db()` → float (current effective SINR in dB)
  - `send_with_mcs(mcs_idx)` → `(ack:int, sinr_eff_db:float)`
- `mcs_table.py` — NR MCS spectral efficiency helper.

## How to use
1. Open this folder in JupyterLab.
2. Open `rl_link_adaptation.ipynb` and run the cells top to bottom.
3. In the “Connect to Sionna tutorial” cell, provide the two functions from your tutorial run:
   ```python
   from la_env import bind_callbacks
   bind_callbacks(get_effective_sinr_db=<YOUR_FUNC>, send_with_mcs=<YOUR_FUNC>)
   ```
   These should come straight from the tutorial (effective SINR calc and PHY abstraction call).
4. Hit Train. The agent will replace ILLA/OLLA and learn from ACK/NACK feedback.

## Notes
- Start simple: state = [sinr_db_norm, last_mcs_norm, last_ack].
- Reward = SE(mcs) - lambda * 1[NACK] tuned by LAMBDA_NACK in the notebook.
- Once stable, extend the state to include SINR/CQI history or BLER moving average.
