# Replays a cached multi-cell route as a per-slot link adaptation environment with HARQ and common random numbers.

import os

import numpy as np
import torch
from sionna.sys import EESM

from . import link
from .mcs import TABLE_INDEX
from .phy import MCS_LIST, NUM_ACTIONS, bler

WINDOW = 8
REPORT_DELAY = 4
REPORT_PERIOD = 10
REPORT_CALIBRATION = os.path.join("cache", "report_error.npz")
BLER_TARGET = 0.1
OBS_DIM = 3 * WINDOW + 2
SLOW_ALPHA = 0.005
HARQ_MAX_ATTEMPTS = 4
HARQ_RTT = 4
CHUNK = 4096
MCS_CATEGORY = 1


class LinkEnv:
    def __init__(self, gains, delays, doppler, slots_per_position, slot_rate):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_cells, self.num_positions, _ = gains.shape
        self.slots_per_position = slots_per_position
        self.slot_period = 1.0 / slot_rate
        self.length = self.num_positions * slots_per_position
        sinr = self._subcarrier_sinr(gains, delays, doppler)
        self.effective_db = self._effective_sinr_db(sinr)
        self.error_rate = np.clip(bler(MCS_LIST[None, :], self.effective_db), 0.0, 1.0)
        self.report_mcs = MCS_LIST[(link.TB_SIZE[MCS_LIST][None, :] * (1.0 - self.error_rate)).argmax(axis=1)]
        self.genie_mcs = self._genie_schedule()
        self.reset(0)

    def _genie_schedule(self):
        linear = 10.0 ** (self.effective_db / 10.0)
        span = HARQ_RTT * HARQ_MAX_ATTEMPTS
        padded = np.vstack([linear, np.repeat(linear[-1:], span, axis=0)])
        accumulated = np.zeros_like(linear)
        survive = np.ones_like(linear)
        expected_bits = np.zeros_like(linear)
        expected_slots = np.zeros_like(linear)
        for attempt in range(HARQ_MAX_ATTEMPTS):
            offset = attempt * HARQ_RTT
            accumulated = accumulated + padded[offset:offset + self.length]
            failure = np.clip(bler(MCS_LIST[None, :], 10.0 * np.log10(np.maximum(accumulated, 1e-12))), 0.0, 1.0)
            success = survive * (1.0 - failure)
            expected_bits = expected_bits + success * link.TB_SIZE[MCS_LIST][None, :]
            expected_slots = expected_slots + success * (attempt + 1)
            survive = survive * failure
        expected_slots = expected_slots + survive * HARQ_MAX_ATTEMPTS
        return MCS_LIST[(expected_bits / expected_slots).argmax(axis=1)]

    def _subcarrier_sinr(self, gains, delays, doppler):
        times = np.arange(self.slots_per_position) * self.slot_period
        grid = np.empty((self.length, link.NUM_SUBCARRIERS), np.float32)
        serving = (np.abs(gains) ** 2).sum(axis=2).argmax(axis=0)
        self.serving_cell = np.repeat(serving, self.slots_per_position)
        for index in range(self.num_positions):
            power = link.cell_powers(gains[:, index], delays[:, index], doppler[:, index], times)
            start = index * self.slots_per_position
            grid[start:start + self.slots_per_position] = link.sinr_from_cells(power, serving[index])
        return grid

    def _effective_sinr_db(self, sinr):
        eesm = EESM()
        table = np.empty((self.length, NUM_ACTIONS), np.float32)
        for action, mcs_index in enumerate(MCS_LIST):
            for start in range(0, self.length, CHUNK):
                block = sinr[start:start + CHUNK]
                tensor = torch.tensor(block[:, None, :, None, None], dtype=torch.float32, device=self.device)
                index = torch.full((block.shape[0], 1), int(mcs_index), dtype=torch.int32, device=self.device)
                effective = eesm(sinr=tensor, mcs_index=index, mcs_table_index=TABLE_INDEX,
                                 mcs_category=MCS_CATEGORY)
                table[start:start + block.shape[0], action] = effective.detach().cpu().numpy().ravel()
        return 10.0 * np.log10(np.maximum(table, 1e-12))

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.uniform = rng.random(self.length)
        reported = self.effective_db[np.arange(self.length), self.report_index()]
        calibration = np.load(REPORT_CALIBRATION)
        bias = np.interp(reported, calibration["bins_db"], calibration["bias_db"])
        spread = np.interp(reported, calibration["bins_db"], calibration["std_db"])
        self.report_db = link.cqi_report(reported, bias + spread * rng.normal(size=self.length))
        self.step_index = 0
        self.mcs_history = np.zeros(WINDOW, np.float32)
        self.ack_history = np.ones(WINDOW, np.float32)
        self.bler_estimate = BLER_TARGET
        self.bler_slow = BLER_TARGET
        self.inflight = []
        self.arrivals = []
        return self.observation()

    def report_index(self):
        return np.searchsorted(MCS_LIST, self.report_mcs)

    def report_slot(self):
        latest = self.step_index - REPORT_DELAY
        return max(0, (latest // REPORT_PERIOD) * REPORT_PERIOD)

    def reported_db(self):
        return float(self.report_db[self.report_slot()])

    def observation(self):
        latest = self.report_slot()
        window = np.clip(latest - np.arange(WINDOW - 1, -1, -1) * REPORT_PERIOD, 0, self.length - 1)
        scaled_cqi = (self.report_db[window] - 10.0) / 10.0
        scaled_mcs = self.mcs_history / float(MCS_LIST[-1])
        return np.concatenate([scaled_cqi, scaled_mcs, self.ack_history,
                               [self.bler_estimate, self.bler_slow]]).astype(np.float32)

    def retransmission_due(self):
        return any(block["ready"] <= self.step_index for block in self.inflight)

    def step(self, mcs_index):
        block = next((entry for entry in self.inflight if entry["ready"] <= self.step_index), None)
        retransmission = block is not None
        if not retransmission:
            block = {"mcs": int(mcs_index), "sinr": 0.0, "attempts": 0, "ready": 0,
                     "first_ack": 0, "bits": 0, "start": self.step_index}
            self.inflight.append(block)
        action = int(np.searchsorted(MCS_LIST, block["mcs"]))
        block["sinr"] += 10.0 ** (self.effective_db[self.step_index, action] / 10.0)
        block["attempts"] += 1
        error_rate = float(bler(block["mcs"], 10.0 * np.log10(max(block["sinr"], 1e-12))))
        ack = int(self.uniform[self.step_index] > error_rate)
        if block["attempts"] == 1:
            block["first_ack"] = ack
        block["ready"] = self.step_index + HARQ_RTT
        if ack or block["attempts"] >= HARQ_MAX_ATTEMPTS:
            block["bits"] = int(link.TB_SIZE[block["mcs"]]) if ack else 0
            self.inflight.remove(block)
            self.arrivals.append(block)
        self.step_index += 1
        delivered = [entry for entry in self.arrivals if entry["ready"] <= self.step_index]
        for entry in delivered:
            self.arrivals.remove(entry)
            self.bler_estimate = 0.95 * self.bler_estimate + 0.05 * (1 - entry["first_ack"])
            self.bler_slow = (1.0 - SLOW_ALPHA) * self.bler_slow + SLOW_ALPHA * (1 - entry["first_ack"])
            self.mcs_history = np.roll(self.mcs_history, -1)
            self.mcs_history[-1] = float(entry["mcs"])
            self.ack_history = np.roll(self.ack_history, -1)
            self.ack_history[-1] = float(entry["first_ack"])
        done = self.step_index >= self.length
        return self.observation(), delivered, retransmission, done


def run_episode(agent, environment, seed):
    observation = environment.reset(seed)
    agent.reset()
    total_bits = 0
    first_acks = []
    choices = []
    retransmissions = 0
    pending = {}
    while True:
        mcs_index = -1
        if not environment.retransmission_due():
            if hasattr(agent, "oracle_mcs"):
                agent.oracle_mcs = int(environment.genie_mcs[environment.step_index])
            slot = environment.step_index
            mcs_index = agent.act(observation, environment.reported_db())
            pending[slot] = (observation, getattr(agent, "last_action", 0))
            choices.append(mcs_index)
        next_observation, delivered, retransmission, done = environment.step(mcs_index)
        if retransmission:
            retransmissions += 1
        for entry in delivered:
            total_bits += entry["bits"]
            first_acks.append(entry["first_ack"])
            state, action = pending.pop(entry["start"])
            reward = entry["bits"] / (entry["attempts"] * float(link.TB_SIZE.max()))
            agent.update(state, action, entry["first_ack"], reward, next_observation, done)
        observation = next_observation
        if done:
            break
    first_acks = np.array(first_acks)
    return {"goodput_mbps": total_bits / (environment.length * environment.slot_period) / 1e6,
            "bler": float(1.0 - first_acks.mean()),
            "retransmission_rate": retransmissions / environment.length,
            "mcs": np.array(choices)}
