# Trains the deep Q network policy on a cached route and writes the checkpoint.

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkadapt import raytrace
from linkadapt.agents import DqnAgent
from linkadapt.env import LinkEnv, run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="san_francisco")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="dqn_policy.pt")
    args = parser.parse_args()

    positions, gains, delays, doppler, slots, rate = raytrace.load(os.path.join("cache", f"{args.scene}.npz"))
    start = time.time()
    environment = LinkEnv(gains, delays, doppler, slots, rate)
    print(f"environment {environment.length} slots, built in {time.time() - start:.1f} s")

    agent = DqnAgent(seed=args.seed, training=True)
    history = []
    best = -1.0
    for episode in range(args.episodes):
        start = time.time()
        result = run_episode(agent, environment, seed=args.seed * 1000 + episode)
        agent.training = False
        greedy = run_episode(agent, environment, seed=99000 + episode)
        agent.training = True
        history.append(greedy["goodput_mbps"])
        marker = ""
        if greedy["goodput_mbps"] > best:
            best = greedy["goodput_mbps"]
            agent.save(args.out)
            marker = " saved"
        print(f"episode {episode + 1:2d}/{args.episodes}  explore {result['goodput_mbps']:6.2f}  "
              f"greedy {greedy['goodput_mbps']:6.2f} Mb/s  BLER {greedy['bler']:.3f}  "
              f"epsilon {agent.epsilon():.3f}  [{time.time() - start:.0f}s]{marker}")
    np.save(os.path.join("results", "tables", f"training_{args.scene}.npy"), np.array(history))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
