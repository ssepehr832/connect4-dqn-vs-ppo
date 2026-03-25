import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .network import ActorCriticNetwork


class RolloutBuffer:
    """Stores one rollout of transitions for on-policy training."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.legal_masks = []

    def push(self, state, action, log_prob, reward, done, value, legal_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.legal_masks.append(legal_mask)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.legal_masks.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO (Proximal Policy Optimization) agent for Connect 4."""

    def __init__(
        self,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        ppo_epochs=4,
        minibatch_size=256,
        rollout_steps=512,
        max_grad_norm=0.5,
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
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm

        self.network = ActorCriticNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.rollout = RolloutBuffer()
        self.steps_done = 0

    def _state_to_tensor(self, state):
        """Convert (6,7,2) numpy state to (1,2,6,7) torch tensor."""
        t = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)

    def _make_legal_mask(self, legal_actions):
        """Create a boolean mask: True for legal actions."""
        mask = torch.zeros(7, dtype=torch.bool, device=self.device)
        for a in legal_actions:
            mask[a] = True
        return mask

    def select_action(self, env, greedy=False):
        """Select action using the policy.

        Args:
            env:    Connect4Env instance
            greedy: if True, pick the highest-probability legal action
        Returns:
            action (int): column index
        """
        legal = env.get_legal_actions()
        state = env.get_state()

        with torch.no_grad():
            logits, _ = self.network(self._state_to_tensor(state))
            logits = logits.squeeze(0)

        # Mask illegal actions
        mask = self._make_legal_mask(legal)
        logits[~mask] = float("-inf")

        if greedy:
            return logits.argmax().item()

        probs = torch.softmax(logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def select_actions_batch(self, states, legal_actions_batch, greedy=False):
        """Batched action selection. Returns actions, log_probs, values, legal_masks.

        Args:
            states:              (N, 6, 7, 2) numpy array
            legal_actions_batch: list of N legal action lists
            greedy:              if True, always pick best action
        Returns:
            actions:     (N,) numpy int array
            log_probs:   (N,) numpy float array
            values:      (N,) numpy float array
            legal_masks: (N, 7) numpy bool array
        """
        n = len(states)
        states_t = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            logits, values = self.network(states_t)  # (N,7), (N,)

        # Build legal mask
        legal_mask = torch.zeros((n, 7), dtype=torch.bool, device=self.device)
        for i, legal in enumerate(legal_actions_batch):
            for a in legal:
                legal_mask[i, a] = True

        logits[~legal_mask] = float("-inf")
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)

        if greedy:
            actions_t = logits.argmax(dim=1)
        else:
            actions_t = dist.sample()

        log_probs_t = dist.log_prob(actions_t)

        return (
            actions_t.cpu().numpy(),
            log_probs_t.cpu().numpy(),
            values.cpu().numpy(),
            legal_mask.cpu().numpy(),
        )

    def compute_gae(self, next_value):
        """Compute GAE advantages and discounted returns from the rollout buffer.

        Args:
            next_value: V(s_{T+1}) — bootstrap value for the last state

        Returns:
            advantages: numpy (T,)
            returns:    numpy (T,)
        """
        rewards = np.array(self.rollout.rewards, dtype=np.float32)
        values = np.array(self.rollout.values, dtype=np.float32)
        dones = np.array(self.rollout.dones, dtype=np.float32)
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_states):
        """Run PPO update on collected rollout.

        Args:
            next_states: (N_ENVS, 6, 7, 2) — states after last step, for bootstrapping

        Returns:
            dict with policy_loss, value_loss, entropy, total_loss
        """
        # Bootstrap value for last state
        with torch.no_grad():
            ns = torch.from_numpy(next_states).float().permute(0, 3, 1, 2).to(self.device)
            _, next_values = self.network(ns)
            # Average across envs as a simple bootstrap
            next_value = next_values.mean().item()

        advantages, returns = self.compute_gae(next_value)

        # Convert rollout to tensors
        states = np.array(self.rollout.states, dtype=np.float32)
        states_t = torch.from_numpy(states).permute(0, 3, 1, 2).to(self.device)
        actions_t = torch.tensor(self.rollout.actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(self.rollout.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.from_numpy(returns).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        legal_masks_t = torch.tensor(np.array(self.rollout.legal_masks), dtype=torch.bool, device=self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        T = len(self.rollout)
        total_stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        for epoch in range(self.ppo_epochs):
            # Shuffle and iterate in minibatches
            indices = np.arange(T)
            np.random.shuffle(indices)

            for start in range(0, T, self.minibatch_size):
                end = min(start + self.minibatch_size, T)
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_legal = legal_masks_t[mb_idx]

                # Forward pass
                logits, values = self.network(mb_states)
                logits[~mb_legal] = float("-inf")
                probs = torch.softmax(logits, dim=1)
                dist = Categorical(probs)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Clipped surrogate loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_stats["policy_loss"] += policy_loss.item()
                total_stats["value_loss"] += value_loss.item()
                total_stats["entropy"] += entropy.item()

        self.steps_done += T
        self.rollout.clear()

        n_updates = self.ppo_epochs * ((T + self.minibatch_size - 1) // self.minibatch_size)
        return {k: v / max(n_updates, 1) for k, v in total_stats.items()}

    def save(self, path):
        """Save model weights and training state."""
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path):
        """Load model weights and training state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
