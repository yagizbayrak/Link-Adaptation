# Writes the inner loop thresholds, the trained network weights and golden vectors in the Q16.16 format the register transfer level code reads.

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkadapt import raytrace
from linkadapt.agents import DqnAgent, OllaAgent, inner_loop_table, inner_loop_mcs
from linkadapt.env import LinkEnv

FRACTION = 16
SCALE = float(1 << FRACTION)
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
ILLA_SAMPLES = 400
OLLA_SAMPLES = 400
DQN_SAMPLES = 40


def quantise(value):
    raw = np.rint(np.asarray(value, np.float64) * SCALE)
    return np.clip(raw, INT32_MIN, INT32_MAX).astype(np.int64)


def write_memory(path, values):
    with open(path, "w") as handle:
        for entry in quantise(values).ravel():
            handle.write(f"{entry & 0xFFFFFFFF:08x}\n")


def main():
    os.makedirs("rtl", exist_ok=True)
    os.makedirs("tb", exist_ok=True)

    edges = np.array(inner_loop_table(), np.float64)
    edges[0] = INT32_MIN / SCALE
    write_memory(os.path.join("rtl", "illa_thresholds.mem"), edges)

    agent = DqnAgent(seed=0, training=False)
    agent.load("dqn_policy.pt")
    weights = [tensor.detach().cpu().numpy() for tensor in agent.online.parameters()]
    for name, array in zip(["w1", "b1", "w2", "b2", "w3", "b3"], weights):
        write_memory(os.path.join("rtl", f"dqn_{name}.mem"), array)

    reports = np.linspace(-30.0, 40.0, ILLA_SAMPLES)
    with open(os.path.join("tb", "vectors_illa.txt"), "w") as handle:
        for report in reports:
            handle.write(f"{quantise(report) & 0xFFFFFFFF:08x} {inner_loop_mcs(report):02d}\n")

    positions, gains, delays, doppler, slots, rate = raytrace.load(os.path.join("cache", "street_canyon.npz"))
    environment = LinkEnv(gains, delays, doppler, slots, rate)

    olla = OllaAgent()
    observation = environment.reset(7)
    olla.reset()
    rng = np.random.default_rng(3)
    with open(os.path.join("tb", "vectors_olla.txt"), "w") as handle:
        for step in range(OLLA_SAMPLES):
            report = float(environment.report_db[step])
            chosen = olla.act(None, report)
            ack = int(rng.random() > 0.15)
            handle.write(f"{quantise(report) & 0xFFFFFFFF:08x} {ack} {chosen:02d}\n")
            olla.update(None, 0, ack, 0.0, None, False)

    observation = environment.reset(11)
    states = []
    reports = []
    decisions = []
    while len(states) < DQN_SAMPLES:
        if not environment.retransmission_due():
            report = environment.reported_db()
            states.append(observation.copy())
            reports.append(report)
            chosen = agent.act(observation, report)
            decisions.append(chosen)
        else:
            chosen = -1
        observation, delivered, retransmission, done = environment.step(chosen)
        if done:
            break
    with open(os.path.join("tb", "vectors_dqn.txt"), "w") as handle:
        for state in states:
            with torch.no_grad():
                tensor = torch.from_numpy(state).to(agent.device).unsqueeze(0)
                action = int(agent.online(tensor).argmax())
            words = " ".join(f"{value & 0xFFFFFFFF:08x}" for value in quantise(state))
            handle.write(f"{words} {action}\n")

    with open(os.path.join("tb", "vectors_dqn_la.txt"), "w") as handle:
        for state, report, chosen in zip(states, reports, decisions):
            words = " ".join(f"{value & 0xFFFFFFFF:08x}" for value in quantise(state))
            handle.write(f"{words} {quantise(report) & 0xFFFFFFFF:08x} {chosen:02d}\n")

    print(f"thresholds {edges.size}, weights {[array.shape for array in weights]}")
    print(f"vectors {ILLA_SAMPLES} illa, {OLLA_SAMPLES} olla, {len(states)} dqn")


if __name__ == "__main__":
    main()
