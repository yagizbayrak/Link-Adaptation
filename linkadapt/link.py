# Builds the per-subcarrier SINR grid from multi-cell ray traced paths and forms the channel quality report.

import numpy as np
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.nr.utils import calculate_tb_size

from . import mcs

SUBCARRIER_SPACING = 30e3
NUM_PRB = 24
NUM_SUBCARRIERS = NUM_PRB * 12
NUM_OFDM_SYMBOLS = 14
CARRIER_PRB = 51
TEMPERATURE = 294.0
NOISE_FIGURE_DB = 9.0
TX_POWER_DBM = 44.0
CQI_MIN_DB = -6.0
CQI_MAX_DB = 24.0
CQI_LEVELS = 16
NUM_DMRS_PER_PRB = 12
NUM_DATA_RE = NUM_PRB * 12 * NUM_OFDM_SYMBOLS - NUM_PRB * NUM_DMRS_PER_PRB


def _transport_block_sizes():
    sizes = []
    for order, efficiency in zip(mcs.MOD_ORDER, mcs.SPECTRAL_EFFICIENCY):
        result = calculate_tb_size(modulation_order=int(order),
                                   target_coderate=float(efficiency) / float(order),
                                   num_prbs=NUM_PRB,
                                   num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                                   num_dmrs_per_prb=NUM_DMRS_PER_PRB,
                                   return_cw_length=False)
        sizes.append(int(result[0]))
    return np.array(sizes, np.int64)


TB_SIZE = _transport_block_sizes()


def subcarrier_offsets():
    index = np.arange(NUM_SUBCARRIERS) - NUM_SUBCARRIERS / 2.0
    return (index * SUBCARRIER_SPACING).astype(np.float64)


def noise_power_watt():
    thermal = BOLTZMANN_CONSTANT * TEMPERATURE * SUBCARRIER_SPACING
    return float(thermal * 10.0 ** (NOISE_FIGURE_DB / 10.0))


def tx_power_per_subcarrier_watt():
    return 10.0 ** ((TX_POWER_DBM - 30.0) / 10.0) / (CARRIER_PRB * 12)


def frequency_response(gains, delays, doppler, times):
    phase = -2.0 * np.pi * np.outer(delays.astype(np.float64), subcarrier_offsets())
    steering = np.exp(1j * phase).astype(np.complex64)
    rotation = np.exp(2j * np.pi * np.outer(times, doppler.astype(np.float64))).astype(np.complex64)
    return (rotation * gains[None, :]) @ steering


def cell_powers(gains, delays, doppler, times):
    power = np.empty((gains.shape[0], times.size, NUM_SUBCARRIERS), np.float32)
    for cell in range(gains.shape[0]):
        response = frequency_response(gains[cell], delays[cell], doppler[cell], times)
        power[cell] = tx_power_per_subcarrier_watt() * np.abs(response) ** 2
    return power


def sinr_from_cells(power, serving):
    signal = power[serving]
    interference = power.sum(axis=0) - signal
    return signal / (interference + noise_power_watt())


def cqi_report(sinr_eff_db, error_db):
    noisy = sinr_eff_db + error_db
    step = (CQI_MAX_DB - CQI_MIN_DB) / (CQI_LEVELS - 1)
    level = np.clip(np.rint((noisy - CQI_MIN_DB) / step), 0, CQI_LEVELS - 1)
    return (CQI_MIN_DB + level * step).astype(np.float32)
