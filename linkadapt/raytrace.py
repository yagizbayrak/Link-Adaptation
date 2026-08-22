# Traces a receiver route against every base station in a Sionna RT scene and caches the path coefficients.

import numpy as np
import mitsuba as mi

if mi.variant() is None:
    mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

SCENES = {
    "san_francisco": rt.scene.san_francisco,
    "munich": rt.scene.munich,
    "street_canyon": rt.scene.simple_street_canyon,
}
CHUNK = 64


def build_scene(scene_name, frequency, tx_positions, look_at):
    scene = load_scene(SCENES[scene_name])
    scene.frequency = frequency
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for index, position in enumerate(tx_positions):
        scene.add(Transmitter(f"tx{index}", position=[float(v) for v in position],
                              look_at=[float(v) for v in look_at]))
    return scene


def surface_heights(scene, x, y):
    x = np.asarray(x, np.float32).ravel()
    y = np.asarray(y, np.float32).ravel()
    below = mi.Point3f(x, y, np.full(x.size, -1000.0, np.float32))
    above = mi.Point3f(x, y, np.full(x.size, 1000.0, np.float32))
    ground = scene.mi_scene.ray_intersect(mi.Ray3f(below, mi.Vector3f(0.0, 0.0, 1.0)))
    roof = scene.mi_scene.ray_intersect(mi.Ray3f(above, mi.Vector3f(0.0, 0.0, -1.0)))
    return np.array(ground.p.z), np.array(roof.p.z)


def street_mask(scene, x, y):
    ground, roof = surface_heights(scene, x, y)
    return np.abs(roof - ground) < 1.0


def route_positions(scene, start, end, num_points, height):
    fraction = np.linspace(0.0, 1.0, num_points)
    x = start[0] + fraction * (end[0] - start[0])
    y = start[1] + fraction * (end[1] - start[1])
    ground, roof = surface_heights(scene, x, y)
    positions = np.stack([x, y, ground + height], axis=1).astype(np.float32)
    return positions, np.abs(roof - ground) < 1.0


def trace_route(scene, positions, speed, max_depth):
    heading = positions[-1, :2] - positions[0, :2]
    heading = heading / np.linalg.norm(heading)
    velocity = [float(heading[0] * speed), float(heading[1] * speed), 0.0]
    solver = PathSolver()
    blocks = []
    for begin in range(0, len(positions), CHUNK):
        names = []
        for offset, position in enumerate(positions[begin:begin + CHUNK]):
            name = f"rx{begin + offset}"
            scene.add(Receiver(name, position=[float(v) for v in position], velocity=velocity))
            names.append(name)
        paths = solver(scene=scene, max_depth=max_depth)
        amplitude, delay = paths.cir(normalize_delays=False, out_type="numpy")
        blocks.append((np.squeeze(amplitude, axis=(1, 3, 5)).astype(np.complex64),
                       delay.astype(np.float32),
                       np.array(paths.doppler, np.float32)))
        for name in names:
            scene.remove(name)
    width = max(block[0].shape[2] for block in blocks)
    shape = (len(positions), blocks[0][0].shape[1], width)
    gains = np.zeros(shape, np.complex64)
    delays = np.zeros(shape, np.float32)
    doppler = np.zeros(shape, np.float32)
    row = 0
    for gain, delay, shift in blocks:
        rows = gain.shape[0]
        gains[row:row + rows, :, :gain.shape[2]] = gain
        delays[row:row + rows, :, :delay.shape[2]] = delay
        doppler[row:row + rows, :, :shift.shape[2]] = shift
        row += rows
    return np.moveaxis(gains, 1, 0), np.moveaxis(delays, 1, 0), np.moveaxis(doppler, 1, 0)


def save(path, positions, gains, delays, doppler, slots_per_position, slot_rate):
    np.savez_compressed(path, positions=positions, gains=gains, delays=delays, doppler=doppler,
                        slots_per_position=slots_per_position, slot_rate=slot_rate)


def load(path):
    data = np.load(path)
    return (data["positions"], data["gains"], data["delays"], data["doppler"],
            int(data["slots_per_position"]), float(data["slot_rate"]))
