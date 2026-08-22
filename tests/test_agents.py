# Checks the memoised baselines reproduce Sionna's own link adaptation decisions exactly.

import numpy as np
import torch
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation, OuterLoopLinkAdaptation

from linkadapt import link
from linkadapt.mcs import TABLE_INDEX
from linkadapt.agents import IllaAgent, OllaAgent
from linkadapt.env import BLER_TARGET, MCS_CATEGORY
from linkadapt.phy import MCS_LIST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPORTS = np.arange(-8.0, 28.0, 0.37)


def allocated():
    return torch.tensor([link.NUM_DATA_RE], dtype=torch.int32, device=DEVICE)


def linear(decibels):
    return torch.tensor([10.0 ** (decibels / 10.0)], dtype=torch.float32, device=DEVICE)


def test_inner_loop_matches_sionna():
    reference = InnerLoopLinkAdaptation(PHYAbstraction(), bler_target=BLER_TARGET)
    agent = IllaAgent()
    for decibels in REPORTS:
        expected = int(reference(sinr_eff=linear(decibels), num_allocated_re=allocated(),
                                 mcs_table_index=TABLE_INDEX, mcs_category=MCS_CATEGORY))
        assert agent.act(None, float(decibels)) == expected


def test_outer_loop_matches_sionna():
    agent = OllaAgent()
    reference = OuterLoopLinkAdaptation(PHYAbstraction(), num_ut=1, bler_target=BLER_TARGET,
                                        delta_up=agent.policy.delta_up)
    agent.reset()
    rng = np.random.default_rng(0)
    feedback = torch.tensor([-1], dtype=torch.int32, device=DEVICE)
    for step in range(200):
        decibels = float(REPORTS[step % REPORTS.size])
        expected = int(reference(num_allocated_re=allocated(), harq_feedback=feedback,
                                 sinr_eff=linear(decibels), mcs_table_index=TABLE_INDEX,
                                 mcs_category=MCS_CATEGORY))
        assert agent.act(None, decibels) == expected
        ack = int(rng.random() > 0.2)
        agent.update(None, expected, ack, 0.0, None, False)
        feedback = torch.tensor([ack], dtype=torch.int32, device=DEVICE)


def test_action_space_is_supported():
    assert MCS_LIST.min() == 3
    assert MCS_LIST.max() == 28
    assert link.TB_SIZE[MCS_LIST].min() > 0
