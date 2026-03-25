import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork
from .replay_buffer import ReplayBuffer, NStepBuffer


class DQNAgent:
    """DQN agent for Connect 4 with experience replay and target network."""

    def __init__(
        self,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=80_000,
        buffer_capacity=100_000,
        batch_size=64,
        target_update_freq=1000,
        n_steps=3,
        n_envs=16,
        device=None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule (linear decay)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.n_steps = n_steps
        self.n_step_buffer = NStepBuffer(n_envs, n_steps, gamma, self.replay_buffer)

        self.steps_done = 0

    @property
    def epsilon(self):
        """Current exploration rate."""
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _state_to_tensor(self, state):
        """Convert (6,7,2) numpy state to (1,2,6,7) torch tensor."""
        t = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)

    def select_action(self, env, greedy=False):
        """Epsilon-greedy action selection with illegal action masking.

        Args:
            env:    Connect4Env instance
            greedy: if True, always pick best action (no exploration)
        Returns:
            action (int): column index
        """
        legal = env.get_legal_actions()

        if not greedy and random.random() < self.epsilon:
            return random.choice(legal)

        state = env.get_state()
        with torch.no_grad():
            q_values = self.q_net(self._state_to_tensor(state)).squeeze(0)

        # Mask illegal actions
        mask = torch.full((7,), float("-inf"), device=self.device)
        for a in legal:
            mask[a] = 0.0
        q_values = q_values + mask

        return q_values.argmax().item()

    def select_actions_batch(self, states, legal_actions_batch, greedy=False):
        """Batched epsilon-greedy action selection.

        Args:
            states:              (N, 6, 7, 2) numpy array
            legal_actions_batch: list of N legal action lists
            greedy:              if True, no exploration
        Returns:
            actions: numpy array of N column indices
        """
        n = len(states)
        actions = np.empty(n, dtype=np.int64)

        # Batched forward pass for all envs
        states_t = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(states_t)  # (N, 7)

        # Build mask: -inf for illegal actions
        mask = torch.full((n, 7), float("-inf"), device=self.device)
        for i, legal in enumerate(legal_actions_batch):
            for a in legal:
                mask[i, a] = 0.0
        q_values = q_values + mask
        best_actions = q_values.argmax(dim=1).cpu().numpy()

        for i in range(n):
            if not greedy and random.random() < self.epsilon:
                actions[i] = random.choice(legal_actions_batch[i])
            else:
                actions[i] = best_actions[i]

        return actions

    def store_transition(self, state, action, reward, next_state, done, next_legal, env_id=0):
        """Store a transition via the n-step buffer."""
        self.n_step_buffer.push(env_id, state, action, reward, next_state, done, next_legal)

    def update(self):
        """Sample a batch and perform one gradient step.

        Returns:
            loss (float) or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_legals = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert to tensors — states are (B, 6, 7, 2) -> (B, 2, 6, 7)
        states_t = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(self.device)
        next_states_t = torch.from_numpy(next_states).float().permute(0, 3, 1, 2).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values with illegal action masking
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            # Mask illegal actions in next states (skip terminal states)
            for i, legal in enumerate(next_legals):
                if legal:  # non-terminal
                    mask = torch.full((7,), float("-inf"), device=self.device)
                    for a in legal:
                        mask[a] = 0.0
                    next_q[i] += mask
            max_next_q = next_q.max(dim=1).values
            # Zero out next-Q for terminal states to avoid NaN from 0 * (-inf)
            max_next_q[dones_t.bool()] = 0.0
            # rewards already contain n-step discounted return R_n,
            # so bootstrap with γ^n (not γ)
            gamma_n = self.gamma ** self.n_steps
            targets = rewards_t + (1.0 - dones_t) * gamma_n * max_next_q

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        """Save model weights and training state."""
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path):
        """Load model weights and training state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
