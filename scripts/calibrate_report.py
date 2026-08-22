# Measures the channel quality report error produced by real DMRS least squares channel estimation.

import argparse
import os
import sys

import numpy as np
import torch
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (LMMSEPostEqualizationSINR, LSChannelEstimator,
                             ResourceGrid, ResourceGridMapper)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkadapt import link, raytrace

BINS_DB = np.arange(-12.0, 32.0, 2.0)
TRIALS = 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="san_francisco")
    parser.add_argument("--positions", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = ResourceGrid(num_ofdm_symbols=link.NUM_OFDM_SYMBOLS, fft_size=link.NUM_SUBCARRIERS,
                        subcarrier_spacing=link.SUBCARRIER_SPACING, num_tx=1, num_streams_per_tx=1,
                        pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2])
    management = StreamManagement(np.array([[1]]), 1)
    mapper = ResourceGridMapper(grid)
    source = BinarySource()
    modulator = Mapper("qam", 4)
    estimator = LSChannelEstimator(grid, interpolation_type="lin")
    post_sinr = LMMSEPostEqualizationSINR(grid, management)

    positions, gains, delays, doppler, slots, rate = raytrace.load(os.path.join("cache", f"{args.scene}.npz"))
    serving = (np.abs(gains) ** 2).sum(axis=2).argmax(axis=0)
    step = max(1, gains.shape[1] // args.positions)
    times = np.zeros(1)
    truth = []
    error = []
    for index in range(0, gains.shape[1], step):
        cell = serving[index]
        power = link.cell_powers(gains[:, index], delays[:, index], doppler[:, index], times)
        interference = float((power.sum(axis=0) - power[cell]).mean()) + link.noise_power_watt()
        response = link.frequency_response(gains[cell, index], delays[cell, index],
                                           doppler[cell, index], times)[0]
        scale = np.sqrt(link.tx_power_per_subcarrier_watt() / interference)
        channel = torch.tensor(np.tile((response * scale)[None, :], (link.NUM_OFDM_SYMBOLS, 1))
                               [None, None, None, None, None], dtype=torch.complex64, device=device)
        noise = torch.tensor(1.0, device=device)
        symbols = mapper(modulator(source([TRIALS, 1, 1, grid.num_data_symbols * 4]))).to(device)
        batch = channel.expand(TRIALS, -1, -1, -1, -1, -1, -1)
        received = (batch * symbols[:, None, None]).sum(dim=(3, 4))
        received = received + torch.sqrt(noise / 2) * (torch.randn_like(received.real)
                                                       + 1j * torch.randn_like(received.real))
        estimate, _ = estimator(received, noise)
        true_db = 10.0 * np.log10(float(post_sinr(channel, noise).mean()))
        for trial in range(TRIALS):
            reported = post_sinr(estimate[trial:trial + 1], noise)
            truth.append(true_db)
            error.append(10.0 * np.log10(float(reported.mean())) - true_db)

    truth = np.array(truth)
    error = np.array(error)
    bias = np.zeros(BINS_DB.size, np.float32)
    spread = np.zeros(BINS_DB.size, np.float32)
    print(f"{'SINR bin':>10} {'count':>7} {'bias dB':>9} {'std dB':>8}")
    for slot in range(BINS_DB.size):
        low = BINS_DB[slot]
        high = low + 2.0
        mask = (truth >= low) & (truth < high)
        if mask.sum() >= 8:
            bias[slot] = error[mask].mean()
            spread[slot] = error[mask].std()
            print(f"{low:6.0f}..{high:<3.0f} {int(mask.sum()):7d} {bias[slot]:9.2f} {spread[slot]:8.2f}")
    valid = spread > 0
    bias = np.interp(BINS_DB, BINS_DB[valid], bias[valid])
    spread = np.interp(BINS_DB, BINS_DB[valid], spread[valid])
    path = os.path.join("cache", "report_error.npz")
    np.savez(path, bins_db=BINS_DB, bias_db=bias, std_db=spread)
    print(f"\noverall bias {error.mean():+.2f} dB, std {error.std():.2f} dB over {error.size} trials")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
