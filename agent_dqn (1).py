import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Replay:
    def __init__(self, capacity:int):
        self.buf = deque(maxlen=capacity)
    def __len__(self): return len(self.buf)
    def add(self, s,a,r,sp,done):
        self.buf.append((s,a,r,sp,done))
    def sample(self, batch_size:int):
        batch = random.sample(self.buf, batch_size)
        s,a,r,sp,d = map(np.array, zip(*batch))
        return s.astype(np.float32), a.astype(np.int32), r.astype(np.float32), sp.astype(np.float32), d.astype(np.float32)

def build_q_net(state_dim:int, num_actions:int, lr:float=1e-3):
    inp = layers.Input((state_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_actions, activation=None)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    return model

class DQNAgent:
    def __init__(self, state_dim:int, num_actions:int,
                 gamma:float=0.99, lr:float=1e-3,
                 eps_start:float=1.0, eps_end:float=0.05, eps_decay:float=2e-4,
                 replay_capacity:int=100000, batch_size:int=256,
                 target_sync:int=1000, seed:int=0):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.rng = np.random.default_rng(seed)

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.replay = Replay(replay_capacity)
        self.q = build_q_net(state_dim, num_actions, lr)
        self.qt = build_q_net(state_dim, num_actions, lr)
        self.qt.set_weights(self.q.get_weights())

        self._steps = 0

    def select_action(self, state:np.ndarray)->int:
        # epsilon-greedy
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.num_actions))
        qvals = self.q.predict(state[None,:].astype(np.float32), verbose=0)[0]
        return int(np.argmax(qvals))

    def push(self, s,a,r,sp,done):
        self.replay.add(s,a,r,sp,done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return False
        s,a,r,sp,d = self.replay.sample(self.batch_size)
        q_cur = self.q.predict(s, verbose=0)
        q_next = self.qt.predict(sp, verbose=0)
        max_next = np.max(q_next, axis=1)
        y = q_cur.copy()
        y[np.arange(self.batch_size), a] = r + (1.0 - d)*self.gamma*max_next
        self.q.train_on_batch(s, y)

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.qt.set_weights(self.q.get_weights())
        # decay epsilon
        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        return True

    def save(self, path:str):
        self.q.save(path)

    def load(self, path:str):
        self.q = keras.models.load_model(path)
        self.qt.set_weights(self.q.get_weights())
