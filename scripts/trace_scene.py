# Ray traces the receiver route against every base station and writes the cached path coefficients.

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkadapt import raytrace

ROUTES = {
    "san_francisco": {"start": (0.0, -75.7), "end": (300.0, -75.7),
                      "sites": [(250.0, -40.0, 31.0), (20.0, -200.0, 55.9),
                                (160.0, 20.0, 32.4), (340.0, -200.0, 49.6)]},
    "munich": {"start": (-26.0, 88.9), "end": (175.0, 88.9),
               "sites": [(120.0, 40.0, 29.0), (-60.0, 180.0, 40.0),
                         (60.0, -30.0, 35.0), (230.0, 150.0, 38.0)]},
    "street_canyon": {"start": (-80.0, 0.0), "end": (80.0, 0.0),
                      "sites": [(-32.0, 10.0, 32.0), (60.0, -40.0, 28.0)]},
}
FREQUENCY = 3.5e9
UE_HEIGHT = 1.7
SPEED = 5.0
MAX_DEPTH = 5
SLOT_RATE = 2000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="san_francisco", choices=sorted(ROUTES))
    parser.add_argument("--positions", type=int, default=512)
    args = parser.parse_args()

    route = ROUTES[args.scene]
    midpoint = ((route["start"][0] + route["end"][0]) / 2.0,
                (route["start"][1] + route["end"][1]) / 2.0, UE_HEIGHT)
    scene = raytrace.build_scene(args.scene, FREQUENCY, route["sites"], midpoint)
    positions, on_street = raytrace.route_positions(scene, route["start"], route["end"],
                                                    args.positions, UE_HEIGHT)
    print(f"scene {args.scene}: {args.positions} positions, on street {on_street.mean() * 100:.1f}%, "
          f"{len(route['sites'])} base stations")
    if on_street.mean() < 1.0:
        print("warning: route leaves the street")
    length = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    slots = int(round(length / (args.positions - 1) / (SPEED / SLOT_RATE)))
    print(f"spacing {length / (args.positions - 1):.3f} m at {SPEED} m/s -> {slots} slots per position")

    start = time.time()
    gains, delays, doppler = raytrace.trace_route(scene, positions, SPEED, MAX_DEPTH)
    print(f"traced in {time.time() - start:.1f} s, gains {gains.shape}")

    path = os.path.join("cache", f"{args.scene}.npz")
    raytrace.save(path, positions, gains, delays, doppler, slots, SLOT_RATE)
    power = (np.abs(gains) ** 2).sum(axis=2)
    for cell in range(gains.shape[0]):
        live = power[cell][power[cell] > 0]
        print(f"  cell {cell}: path gain {10 * np.log10(live.min()):.1f} .. "
              f"{10 * np.log10(live.max()):.1f} dB")
    serving = power.argmax(axis=0)
    print(f"serving cell changes {int((np.diff(serving) != 0).sum())} times along the route")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB), "
          f"{args.positions * slots} slots")


if __name__ == "__main__":
    main()
