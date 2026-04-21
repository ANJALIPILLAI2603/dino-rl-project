# agent_q.py
"""
Q-Learning Agent  (Off-Policy TD Control)
==========================================
Maintains a Q-table: Q[state, action] = expected cumulative reward.

Update rule (Bellman equation):
    Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a')  −  Q(s, a) ]

Key properties:
  • Off-policy : learns the optimal policy even while exploring (greedy target)
  • Tabular    : discrete states only — fast and easy to inspect
"""

import numpy as np
import os
import pickle


class QLearningAgent:
    N_STATES  = 22   # 11 distance buckets × 2 jump states  (see get_discrete_state)
    N_ACTIONS =  2   # 0 = do nothing, 1 = jump

    def __init__(
        self,
        alpha        = 0.15,   # Learning rate      — how fast to update Q-values
        gamma        = 0.99,   # Discount factor    — how much future rewards matter
        epsilon      = 1.0,    # Exploration rate   — probability of random action
        epsilon_min  = 0.01,   # Minimum epsilon after decay
        epsilon_decay= 0.997,  # Per-episode multiplicative decay
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: rows = states (0-21), cols = actions (0-1)
        # Initialised to zero — agent starts with no knowledge
        self.q_table = np.zeros((self.N_STATES, self.N_ACTIONS))

    # ─────────────────────────────────────────────────────────────────────

    def choose_action(self, state):
        """
        Epsilon-greedy policy:
          • With probability ε  → random action  (exploration)
          • Otherwise           → best-known action  (exploitation)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)     # Explore
        return int(np.argmax(self.q_table[state]))       # Exploit

    def update(self, state, action, reward, next_state, done):
        """
        Apply a single Q-Learning update.

        Parameters
        ----------
        state, action   : the transition that just happened
        reward          : reward received
        next_state      : state we landed in
        done            : whether the episode ended
        """
        current_q = self.q_table[state, action]

        if done:
            # No future rewards after terminal state
            target = reward
        else:
            # Greedy look-ahead — take the best possible action from next_state
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Nudge the current Q-value towards the target
        self.q_table[state, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Reduce exploration rate.  Call once at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path="models/q_learning.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"[Q-Learning] Saved → {path}")

    def load(self, path="models/q_learning.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"[Q-Learning] Loaded ← {path}")
