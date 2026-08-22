# Block error rate lookup table built once from Sionna's physical layer abstraction.

import numpy as np
import torch
from sionna.sys import PHYAbstraction

from .mcs import NUM_MCS, TABLE_INDEX

SNR_MIN_DB = -5.0
SNR_MAX_DB = 30.0
SNR_STEP_DB = 0.1
CODE_BLOCK_SIZE = 8448
MCS_CATEGORY = 1
SNR_GRID_DB = np.arange(SNR_MIN_DB, SNR_MAX_DB + SNR_STEP_DB, SNR_STEP_DB, dtype=np.float32)


def _build_table():
    abstraction = PHYAbstraction()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = torch.tensor(10.0 ** (SNR_GRID_DB / 10.0), dtype=torch.float32, device=device)
    table = np.ones((NUM_MCS, SNR_GRID_DB.size), np.float32)
    supported = []
    for mcs in range(NUM_MCS):
        values = abstraction.get_bler(mcs, TABLE_INDEX, MCS_CATEGORY, CODE_BLOCK_SIZE, linear).cpu().numpy()
        if np.isfinite(values).all():
            table[mcs] = np.clip(values, 0.0, 1.0)
            supported.append(mcs)
    return table, np.array(supported, np.int32)


BLER_TABLE, MCS_LIST = _build_table()
NUM_ACTIONS = int(MCS_LIST.size)


def snr_index(sinr_db):
    raw = (np.asarray(sinr_db, np.float32) - SNR_MIN_DB) / SNR_STEP_DB
    return np.clip(np.rint(raw), 0, SNR_GRID_DB.size - 1).astype(np.int64)


def bler(mcs, sinr_db):
    return BLER_TABLE[np.asarray(mcs, np.int64), snr_index(sinr_db)]


def highest_mcs(sinr_db, target):
    column = BLER_TABLE[MCS_LIST, snr_index(sinr_db)]
    feasible = np.nonzero(column <= target)[0]
    return int(MCS_LIST[feasible[-1]]) if feasible.size else int(MCS_LIST[0])
