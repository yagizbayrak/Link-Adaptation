# Evaluates every policy on a scene over several seeds and writes the results table.

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkadapt import raytrace
from linkadapt.agents import DqnAgent, GenieAgent, IllaAgent, OllaAgent
from linkadapt.env import LinkEnv, run_episode

SEEDS = (1, 2, 3, 4, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="san_francisco")
    parser.add_argument("--policy", default="dqn_policy.pt")
    args = parser.parse_args()

    positions, gains, delays, doppler, slots, rate = raytrace.load(os.path.join("cache", f"{args.scene}.npz"))
    environment = LinkEnv(gains, delays, doppler, slots, rate)
    print(f"{args.scene}: {environment.length} slots, {environment.num_cells} cells")

    learned = DqnAgent(seed=0, training=False)
    learned.load(args.policy)
    policies = [("ILLA", IllaAgent()), ("OLLA", OllaAgent()),
                ("DQN", learned), ("Genie", GenieAgent())]

    rows = []
    traces = {}
    for name, agent in policies:
        for seed in SEEDS:
            result = run_episode(agent, environment, seed=seed)
            rows.append({"agent": name, "seed": seed,
                         "goodput_mbps": result["goodput_mbps"],
                         "bler": result["bler"],
                         "retransmission_rate": result["retransmission_rate"],
                         "mean_mcs": float(result["mcs"].mean())})
            if seed == SEEDS[0]:
                traces[name] = result["mcs"]
    frame = pd.DataFrame(rows)
    summary = frame.groupby("agent").agg(["mean", "std"]).round(4)

    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    frame.to_csv(os.path.join("results", "tables", f"episodes_{args.scene}.csv"), index=False)
    summary.to_csv(os.path.join("results", "tables", f"summary_{args.scene}.csv"))
    np.savez(os.path.join("results", "tables", f"traces_{args.scene}.npz"),
             effective_db=environment.effective_db[np.arange(environment.length), environment.report_index()],
             report_db=environment.report_db, serving=environment.serving_cell,
             genie_mcs=environment.genie_mcs, **traces)

    print(f"\n{'agent':7s} {'goodput Mb/s':>16} {'BLER':>14} {'retx':>10} {'meanMCS':>9}")
    for name, _ in policies:
        part = frame[frame.agent == name]
        print(f"{name:7s} {part.goodput_mbps.mean():10.2f} +-{part.goodput_mbps.std():5.2f} "
              f"{part.bler.mean():9.3f} +-{part.bler.std():5.3f} "
              f"{part.retransmission_rate.mean() * 100:8.1f}% {part.mean_mcs.mean():9.1f}")
    baseline = frame[frame.agent == "ILLA"].goodput_mbps.mean()
    ceiling = frame[frame.agent == "Genie"].goodput_mbps.mean()
    for name in ("OLLA", "DQN"):
        value = frame[frame.agent == name].goodput_mbps.mean()
        print(f"{name} closes {(value - baseline) / (ceiling - baseline) * 100:.0f}% of the ILLA to Genie gap")


if __name__ == "__main__":
    main()
