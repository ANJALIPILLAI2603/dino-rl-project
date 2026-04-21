# agent_sarsa.py
"""
SARSA Agent  (On-Policy TD Control)
=====================================
Very similar to Q-Learning, but uses the action *actually chosen* for next
state rather than the greedy maximum.

Update rule:
    Q(s, a) ← Q(s, a) + α · [ r + γ · Q(s', a')  −  Q(s, a) ]

where a' is sampled from the same epsilon-greedy policy — NOT argmax.

Key difference from Q-Learning:
  • On-policy  : the update reflects the exploration policy itself
  • Tends to learn slightly more conservative behaviour (safer near edges)
  • Needs the next action to be chosen *before* calling update()
"""

import numpy as np
import os
import pickle


class SARSAAgent:
    N_STATES  = 22
    N_ACTIONS =  2

    def __init__(
        self,
        alpha        = 0.15,
        gamma        = 0.99,
        epsilon      = 1.0,
        epsilon_min  = 0.01,
        epsilon_decay= 0.997,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((self.N_STATES, self.N_ACTIONS))

    # ─────────────────────────────────────────────────────────────────────

    def choose_action(self, state):
        """
        Epsilon-greedy policy — same as Q-Learning.
        The crucial difference is that SARSA calls this to pick the *next*
        action *before* the update, ensuring the update is on-policy.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update — uses the actual next action (on-policy).

        Parameters
        ----------
        state, action      : transition taken
        reward             : reward received
        next_state         : resulting state
        next_action        : action *already chosen* for next_state (on-policy)
        done               : episode terminated?
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            # On-policy: use Q of the action we *will* take, not the greedy max
            target = reward + self.gamma * self.q_table[next_state, next_action]

        self.q_table[state, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Call at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ─────────────────────────────────────────────────────────────────────

    def save(self, path="models/sarsa.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"[SARSA]      Saved → {path}")

    def load(self, path="models/sarsa.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"[SARSA]      Loaded ← {path}")
