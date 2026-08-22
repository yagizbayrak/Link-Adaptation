# Renders the scene geometry with the base stations and the receiver route marked.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkadapt import raytrace
from sionna.rt import Camera, Receiver
from trace_scene import FREQUENCY, ROUTES, UE_HEIGHT

CAMERAS = {
    "san_francisco": {"position": (150.0, -650.0, 420.0), "look_at": (150.0, -60.0, 20.0)},
    "munich": {"position": (75.0, -420.0, 380.0), "look_at": (75.0, 90.0, 20.0)},
    "street_canyon": {"position": (0.0, -230.0, 130.0), "look_at": (0.0, 0.0, 5.0)},
}
MARKERS = 40


def main():
    scene_name = sys.argv[1] if len(sys.argv) > 1 else "san_francisco"
    route = ROUTES[scene_name]
    midpoint = ((route["start"][0] + route["end"][0]) / 2.0,
                (route["start"][1] + route["end"][1]) / 2.0, UE_HEIGHT)
    scene = raytrace.build_scene(scene_name, FREQUENCY, route["sites"], midpoint)
    positions, _ = raytrace.route_positions(scene, route["start"], route["end"], MARKERS, UE_HEIGHT)
    for index, position in enumerate(positions):
        scene.add(Receiver(f"ue{index}", position=[float(v) for v in position]))
    view = CAMERAS[scene_name]
    camera = Camera(position=view["position"], look_at=view["look_at"])
    target = os.path.join("results", "figures", f"scene_{scene_name}.png")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    scene.render_to_file(camera=camera, filename=target, resolution=(1100, 620),
                         num_samples=192, lighting_scale=1.6)
    print(f"wrote {target} ({os.path.getsize(target) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
