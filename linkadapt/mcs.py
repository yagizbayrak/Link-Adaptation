# 3GPP TS 38.214 modulation and coding scheme table, read from Sionna rather than retyped.

import numpy as np
import torch
from sionna.phy.nr.utils import decode_mcs_index

TABLE_INDEX = 1


def _read_table():
    orders = []
    efficiencies = []
    for index in range(32):
        try:
            order, rate = decode_mcs_index(torch.tensor(index), table_index=TABLE_INDEX, is_pusch=False)
        except Exception:
            break
        orders.append(int(order))
        efficiencies.append(float(order) * float(rate))
    return np.array(orders, np.int32), np.array(efficiencies, np.float32)


MOD_ORDER, SPECTRAL_EFFICIENCY = _read_table()
NUM_MCS = int(len(SPECTRAL_EFFICIENCY))
