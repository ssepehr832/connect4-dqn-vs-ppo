import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer storing (s, a, r, s', done, legal') tuples."""

    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, next_legal):
        """Store a transition."""
        item = (state, action, reward, next_state, done, next_legal)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a random batch and return numpy arrays / lists."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_legals = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            list(next_legals),
        )

    def __len__(self):
        return len(self.buffer)


class NStepBuffer:
    """Accumulates n-step returns before pushing to the main replay buffer.

    For each env, maintains a staging deque. When it has n transitions
    (or hits a terminal state), computes:
        R_n = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(n-1)*r_{n-1}
    and stores (s_0, a_0, R_n, s_n, done_n, legal_n) into the replay buffer.
    """

    def __init__(self, n_envs, n_steps, gamma, replay_buffer):
        self.n_steps = n_steps
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.staging = [deque(maxlen=n_steps) for _ in range(n_envs)]

    def push(self, env_id, state, action, reward, next_state, done, next_legal):
        """Add a transition for a specific env."""
        self.staging[env_id].append((state, action, reward, next_state, done, next_legal))

        if done:
            # Flush all accumulated transitions on terminal
            self._flush(env_id)
        elif len(self.staging[env_id]) == self.n_steps:
            # We have n steps — push the oldest one with n-step return
            self._push_nstep(env_id)

    def _push_nstep(self, env_id):
        """Compute n-step return from full staging buffer and push oldest transition."""
        buf = self.staging[env_id]
        # n-step discounted return
        R = 0.0
        for i in reversed(range(len(buf))):
            R = buf[i][2] + self.gamma * R

        # Use s_0, a_0 from the oldest; s_n, done_n, legal_n from the newest
        s0, a0 = buf[0][0], buf[0][1]
        _, _, _, s_n, done_n, legal_n = buf[-1]

        self.replay_buffer.push(s0, a0, R, s_n, done_n, legal_n)
        # Remove only the oldest — slide the window forward
        buf.popleft()

    def _flush(self, env_id):
        """On terminal: push all remaining transitions with shortened n-step returns."""
        buf = self.staging[env_id]
        while buf:
            R = 0.0
            for i in reversed(range(len(buf))):
                R = buf[i][2] + self.gamma * R

            s0, a0 = buf[0][0], buf[0][1]
            _, _, _, s_n, done_n, legal_n = buf[-1]

            self.replay_buffer.push(s0, a0, R, s_n, done_n, legal_n)
            buf.popleft()
