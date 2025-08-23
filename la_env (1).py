
# Adapter to integrate an RL agent with Sionna's Link Adaptation tutorial.
# This version uses a stateful current effective SINR (dB) and Sionna's PHYAbstraction.get_bler.
#
# You will:
#   1) call set_current_sinr_db(value) inside your simulation loop
#   2) the RL agent calls get_effective_sinr_db() to observe the state
#   3) when the agent chooses an MCS, call send_with_mcs(mcs_idx) to get (ack, sinr_eff_db)
#
# Internals:
#   - We require a PHYAbstraction instance bound via bind_phy_abs(phy_abs).
#   - ACK is drawn as Bernoulli(1 - BLER(mcs, sinr_eff_db)).

from typing import Optional, Tuple
# la_env.py

from functools import lru_cache

import numpy as np

_current_sinr_db = None
_phy_abs = None

# Defaults (all INTs now)
_MCS_TABLE_INDEX = 1   # 1 -> NR MCS Table 1 (up to 64-QAM), 2 -> Table 2 (up to 256-QAM)
_MCS_CATEGORY     = 1   # <-- MUST be INT for your Sionna (not "table1")
_CB_SIZE          = 8448

def bind_phy_abs(phy_abs):
    global _phy_abs
    _phy_abs = phy_abs

def configure_link(mcs_table_index=1, mcs_category=1, cb_size=8448):
    """Call once after bind_phy_abs(...). All args must be ints for your Sionna."""
    global _MCS_TABLE_INDEX, _MCS_CATEGORY, _CB_SIZE
    _MCS_TABLE_INDEX = int(mcs_table_index)
    _MCS_CATEGORY    = int(mcs_category)   # <-- ensure INT
    _CB_SIZE         = int(cb_size)
        # after setting _MCS_TABLE_INDEX, _MCS_CATEGORY, _CB_SIZE:
    try:
        _bler_cached.cache_clear()
    except NameError:
        pass


def set_current_sinr_db(value: float):
    global _current_sinr_db
    _current_sinr_db = float(value)

def get_effective_sinr_db() -> float:
    assert _current_sinr_db is not None, "current SINR not set. Call set_current_sinr_db(...) each step."
    return float(_current_sinr_db)

@lru_cache(maxsize=200_000)
def _bler_cached(mcs_idx: int, snr_tenths: int) -> float:
    # Uses la_env globals set by configure_link() and bind_phy_abs()
    global _phy_abs, _MCS_TABLE_INDEX, _MCS_CATEGORY, _CB_SIZE
    snr_db  = snr_tenths / 10.0
    snr_eff = 10.0**(snr_db/10.0)
    b = float(_phy_abs.get_bler(int(mcs_idx),
                                int(_MCS_TABLE_INDEX),
                                int(_MCS_CATEGORY),
                                int(_CB_SIZE),
                                np.array([snr_eff]))[0])
    if not np.isfinite(b):  # sanitize
        b = 1.0
    return 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)


def send_with_mcs(mcs_idx):
    assert _phy_abs is not None, "PHYAbstraction not bound. Call bind_phy_abs(phy_abs)."
    sinr_eff_db = get_effective_sinr_db()
    snr_tenths  = int(round(sinr_eff_db * 10))
    bler        = _bler_cached(int(mcs_idx), snr_tenths)
    ack         = int(np.random.rand() > bler)  # 1 if success
    return ack, sinr_eff_db

