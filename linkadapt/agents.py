# Link adaptation policies sharing one interface: inner loop, outer loop, deep Q network, and a genie bound.

import numpy as np
import torch
from torch import nn
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation, OuterLoopLinkAdaptation

from . import link
from .mcs import TABLE_INDEX
from .env import BLER_TARGET, MCS_CATEGORY, OBS_DIM
from .phy import MCS_LIST, NUM_ACTIONS

HIDDEN = 256
GAMMA = 0.0
OFFSETS = np.array([-4, -3, -2, -1, 0, 1, 2], np.int32)
NUM_OFFSETS = int(OFFSETS.size)
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
BUFFER_SIZE = 100000
TARGET_SYNC = 500
TRAIN_EVERY = 4
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 30000


GRID_MIN_DB = -40.0
GRID_MAX_DB = 60.0
SEARCH_STEPS = 30
_TABLE = None


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def inner_loop_table():
    global _TABLE
    if _TABLE is None:
        device = _device()
        policy = InnerLoopLinkAdaptation(PHYAbstraction(), bler_target=BLER_TARGET)
        allocated = torch.tensor([link.NUM_DATA_RE], dtype=torch.int32, device=device)

        def decide(decibels):
            linear = torch.tensor([10.0 ** (decibels / 10.0)], dtype=torch.float32, device=device)
            return int(policy(sinr_eff=linear, num_allocated_re=allocated,
                              mcs_table_index=TABLE_INDEX, mcs_category=MCS_CATEGORY))

        edges = np.full(NUM_ACTIONS, -np.inf)
        for action in range(1, NUM_ACTIONS):
            low = GRID_MIN_DB
            high = GRID_MAX_DB
            for _ in range(SEARCH_STEPS):
                middle = 0.5 * (low + high)
                if decide(middle) >= int(MCS_LIST[action]):
                    high = middle
                else:
                    low = middle
            edges[action] = high
        _TABLE = edges
    return _TABLE


def inner_loop_mcs(decibels):
    edges = inner_loop_table()
    return int(MCS_LIST[np.searchsorted(edges, decibels, side="right") - 1])


class IllaAgent:
    def __init__(self):
        inner_loop_table()

    def reset(self):
        return

    def act(self, observation, reported_db):
        return inner_loop_mcs(reported_db)

    def update(self, observation, action, ack, reward, next_observation, done):
        return


class OllaAgent:
    def __init__(self):
        inner_loop_table()
        self.policy = OuterLoopLinkAdaptation(PHYAbstraction(), num_ut=1, bler_target=BLER_TARGET)
        self.offset = 0.0

    def reset(self):
        self.offset = 0.0

    def act(self, observation, reported_db):
        return inner_loop_mcs(reported_db - self.offset)

    def update(self, observation, action, ack, reward, next_observation, done):
        step = -self.policy.delta_down if ack else self.policy.delta_up
        self.offset = float(np.clip(self.offset + step, self.policy.offset_min, self.policy.offset_max))


class GenieAgent:
    def __init__(self):
        self.oracle_mcs = int(MCS_LIST[0])

    def reset(self):
        return

    def act(self, observation, reported_db):
        return self.oracle_mcs

    def update(self, observation, action, ack, reward, next_observation, done):
        return


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
                                    nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                    nn.Linear(HIDDEN, NUM_OFFSETS))

    def forward(self, x):
        return self.layers(x)


class DqnAgent:
    def __init__(self, seed, training):
        self.device = _device()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.states = np.zeros((BUFFER_SIZE, OBS_DIM), np.float32)
        self.next_states = np.zeros((BUFFER_SIZE, OBS_DIM), np.float32)
        self.actions = np.zeros(BUFFER_SIZE, np.int64)
        self.rewards = np.zeros(BUFFER_SIZE, np.float32)
        self.dones = np.zeros(BUFFER_SIZE, np.float32)
        self.written = 0
        self.cursor = 0
        self.steps = 0
        self.training = training
        self.last_action = 0

    def reset(self):
        return

    def epsilon(self):
        if not self.training:
            return 0.0
        decay = np.exp(-self.steps / EPSILON_DECAY)
        return EPSILON_END + (EPSILON_START - EPSILON_END) * decay

    def act(self, observation, reported_db):
        if self.rng.random() < self.epsilon():
            self.last_action = int(self.rng.integers(NUM_OFFSETS))
        else:
            with torch.no_grad():
                state = torch.from_numpy(observation).to(self.device).unsqueeze(0)
                self.last_action = int(self.online(state).argmax())
        base = inner_loop_mcs(reported_db)
        return int(np.clip(base + OFFSETS[self.last_action], MCS_LIST[0], MCS_LIST[-1]))

    def update(self, observation, action, ack, reward, next_observation, done):
        if not self.training:
            return
        slot = self.cursor
        self.states[slot] = observation
        self.next_states[slot] = next_observation
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.dones[slot] = float(done)
        self.cursor = (self.cursor + 1) % BUFFER_SIZE
        self.written = min(self.written + 1, BUFFER_SIZE)
        self.steps += 1
        if self.written < BATCH_SIZE or self.steps % TRAIN_EVERY:
            return
        batch = self.rng.integers(self.written, size=BATCH_SIZE)
        states = torch.from_numpy(self.states[batch]).to(self.device)
        next_states = torch.from_numpy(self.next_states[batch]).to(self.device)
        actions = torch.from_numpy(self.actions[batch]).to(self.device)
        rewards = torch.from_numpy(self.rewards[batch]).to(self.device)
        dones = torch.from_numpy(self.dones[batch]).to(self.device)
        values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.online(next_states).argmax(dim=1, keepdim=True)
            bootstrap = self.target(next_states).gather(1, best).squeeze(1)
            targets = rewards + GAMMA * (1.0 - dones) * bootstrap
        loss = nn.functional.smooth_l1_loss(values, targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        if self.steps % TARGET_SYNC == 0:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
