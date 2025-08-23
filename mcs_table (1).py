import numpy as np

SE_TABLE = np.array([
    0.234, 0.377, 0.601, 0.877, 1.175, 1.476, 1.914, 2.406,
    2.730, 3.322, 3.902, 4.523, 5.115, 5.553, 5.914, 6.226,
    6.402, 6.602, 6.914, 7.115, 7.206, 7.306, 7.356, 7.406,
    7.406, 7.406, 7.406, 7.406
], dtype=np.float32)

NUM_MCS = len(SE_TABLE)

def spectral_efficiency(mcs_idx:int)->float:
    return float(SE_TABLE[int(mcs_idx)])
