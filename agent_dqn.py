# agent_dqn.py
"""
Deep Q-Network (DQN) Agent — pure NumPy implementation
=======================================================
Uses a small neural network instead of a Q-table.
Works with *continuous* state features → can generalise across similar states.

Architecture:
    Input(5) → Dense(32, ReLU) → Dense(32, ReLU) → Output(2)

Key DQN techniques implemented:
  1. Experience Replay   — break correlation by training on random past samples
  2. Target Network      — stable Q-value targets (separate weights, synced slowly)
  3. Epsilon-Greedy      — balance exploration vs exploitation

No external ML libraries required — only NumPy.
"""

import numpy as np
import os
import pickle
from collections import deque


# ══════════════════════════════════════════════════════════════════════════════
# Minimal 2-hidden-layer neural network (pure NumPy)
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    Fully-connected MLP with configurable layer sizes.
    Training uses mini-batch gradient descent with momentum.

    Parameters
    ----------
    layer_sizes : list[int]  e.g. [5, 32, 32, 2]
    lr          : learning rate
    """

    def __init__(self, layer_sizes, lr=0.001):
        self.lr   = lr
        self.beta = 0.9        # Momentum coefficient

        self.weights = []
        self.biases  = []
        self.vW      = []      # Momentum buffers for weights
        self.vb      = []      # Momentum buffers for biases

        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            # He initialisation — good default for ReLU networks
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W);  self.vW.append(np.zeros_like(W))
            self.biases.append(b);   self.vb.append(np.zeros_like(b))

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, X):
        """
        Full forward pass that also caches activations for backprop.
        X shape: (batch_size, input_dim)
        """
        self.cache = [X]                       # cache[0] = input
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            # ReLU on all layers except the output layer
            a = np.maximum(0, z) if i < len(self.weights) - 1 else z
            self.cache.append(a)
        return self.cache[-1]                  # (batch, output_dim)

    def predict(self, x):
        """Single-sample forward pass (no gradient cache needed)."""
        a = x.reshape(1, -1)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = np.maximum(0, z) if i < len(self.weights) - 1 else z
        return a[0]

    # ── Backward pass + weight update ─────────────────────────────────────

    def backward(self, X, y_true):
        """
        Compute gradients via backpropagation and update weights.

        Loss: Mean Squared Error  →  dL/d_out = 2*(y_pred − y_true) / batch
        We use the factor 1/batch only (drop the 2 — absorbed into lr).
        """
        batch  = X.shape[0]
        y_pred = self.forward(X)

        # Gradient of MSE w.r.t. network output
        delta = (y_pred - y_true) / batch     # (batch, out)

        grads_W, grads_b = [], []

        # Iterate layers from last to first
        for i in reversed(range(len(self.weights))):
            A_prev = self.cache[i]             # activation going INTO layer i

            gW = A_prev.T @ delta              # (n_in, n_out)
            gb = delta.sum(axis=0)             # (n_out,)

            grads_W.insert(0, gW)
            grads_b.insert(0, gb)

            if i > 0:
                # Backprop gradient through weights …
                delta = delta @ self.weights[i].T          # (batch, n_in)
                # … and through the ReLU of the previous activation
                delta = delta * (self.cache[i] > 0)

        # SGD with momentum update
        for i in range(len(self.weights)):
            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * grads_W[i]
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * grads_b[i]
            self.weights[i] -= self.lr * self.vW[i]
            self.biases[i]  -= self.lr * self.vb[i]

    def copy_weights_from(self, other):
        """Clone weights from another NeuralNetwork (used for target network)."""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i]  = other.biases[i].copy()


# ══════════════════════════════════════════════════════════════════════════════
# DQN Agent
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    INPUT_SIZE  = 5    # Must match get_continuous_state() output length
    OUTPUT_SIZE = 2    # One Q-value per action

    def __init__(
        self,
        lr                 = 0.001,
        gamma              = 0.99,
        epsilon            = 1.0,
        epsilon_min        = 0.01,
        epsilon_decay      = 0.997,
        buffer_size        = 4000,   # Replay buffer capacity
        batch_size         = 64,     # Mini-batch size for each training step
        target_update_freq = 200,    # Sync target network every N gradient steps
    ):
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._grad_step         = 0    # Count gradient steps for target sync

        # ── Online network — trained every step ──────────────────────────
        arch = [self.INPUT_SIZE, 32, 32, self.OUTPUT_SIZE]
        self.online_net = NeuralNetwork(arch, lr=lr)

        # ── Target network — updated periodically for stable targets ─────
        self.target_net = NeuralNetwork(arch, lr=lr)
        self.target_net.copy_weights_from(self.online_net)

        # ── Experience replay buffer ─────────────────────────────────────
        self.replay_buffer = deque(maxlen=buffer_size)

    # ─────────────────────────────────────────────────────────────────────

    def choose_action(self, state):
        """Epsilon-greedy action selection using the online network."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.OUTPUT_SIZE)       # Explore
        q_vals = self.online_net.predict(np.array(state, dtype=np.float32))
        return int(np.argmax(q_vals))                        # Exploit

    def remember(self, state, action, reward, next_state, done):
        """Store one experience tuple in the replay buffer."""
        self.replay_buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def train_step(self):
        """
        Sample a random mini-batch and perform one gradient update.
        Skipped silently until the buffer has enough experiences.
        """
        if len(self.replay_buffer) < self.batch_size:
            return   # Wait until we have enough data

        # ── Sample random mini-batch from replay buffer ──────────────────
        idx     = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch   = [self.replay_buffer[i] for i in idx]

        states      = np.stack([b[0] for b in batch])          # (B, 5)
        actions     = np.array([b[1] for b in batch])          # (B,)
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)  # (B,)
        next_states = np.stack([b[3] for b in batch])          # (B, 5)
        dones       = np.array([b[4] for b in batch], dtype=np.float32)  # (B,)

        # ── Compute Q-targets using the frozen target network ────────────
        q_next  = np.array([self.target_net.predict(s) for s in next_states])  # (B, 2)
        targets = rewards + self.gamma * np.max(q_next, axis=1) * (1.0 - dones)

        # ── Build the full target matrix — only change the chosen action ──
        q_pred = self.online_net.forward(states)   # (B, 2)
        y      = q_pred.copy()
        y[np.arange(self.batch_size), actions] = targets   # zero-loss on other action

        # ── Gradient update on online network ────────────────────────────
        self.online_net.backward(states, y)

        # ── Periodically sync target network ─────────────────────────────
        self._grad_step += 1
        if self._grad_step % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.online_net)

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ─────────────────────────────────────────────────────────────────────

    def save(self, path="models/dqn.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "weights": self.online_net.weights,
                "biases":  self.online_net.biases,
                "epsilon": self.epsilon,
            }, f)
        print(f"[DQN]        Saved → {path}")

    def load(self, path="models/dqn.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.online_net.weights = data["weights"]
        self.online_net.biases  = data["biases"]
        self.target_net.copy_weights_from(self.online_net)
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"[DQN]        Loaded ← {path}")
