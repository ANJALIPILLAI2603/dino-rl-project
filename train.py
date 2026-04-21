# train.py
"""
Headless Training Loops
========================
Each function trains one agent for N episodes WITHOUT Pygame rendering.
This is faster than visual mode and useful for pre-training before demo.

After training, call agent.save() to persist the Q-table / network weights.
"""

import numpy as np
from environment import DinoEnvironment
from agent_q     import QLearningAgent
from agent_sarsa import SARSAAgent
from agent_dqn   import DQNAgent


def train_q_learning(agent: QLearningAgent, n_episodes=800, verbose=True):
    """
    Train a Q-Learning agent over n_episodes.

    Episode loop:
      1. Reset environment → get initial discrete state
      2. Loop until done:
           a. Choose action (epsilon-greedy)
           b. Step environment → (next_state, reward, done)
           c. Update Q-table
           d. Advance state
      3. Decay epsilon
      4. Log every 100 episodes

    Returns list of per-episode scores (survival frames).
    """
    env    = DinoEnvironment()
    scores = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()

        while not env.done:
            action     = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()
        scores.append(env.score)

        if verbose and ep % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"  [Q-Learning]  ep {ep:4d}  score {env.score:5d}"
                  f"  avg100 {avg:6.0f}  ε={agent.epsilon:.3f}")

    return scores


def train_sarsa(agent: SARSAAgent, n_episodes=800, verbose=True):
    """
    Train a SARSA agent over n_episodes.

    SARSA needs the next action *before* the update call (on-policy).
    We therefore pick action at the start of each step and carry it forward.
    """
    env    = DinoEnvironment()
    scores = []

    for ep in range(1, n_episodes + 1):
        state  = env.reset()
        action = agent.choose_action(state)   # Choose first action up front

        while not env.done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)   # On-policy next

            # SARSA update uses next_action (not greedy max)
            agent.update(state, action, reward, next_state, next_action, done)

            state  = next_state
            action = next_action

        agent.decay_epsilon()
        scores.append(env.score)

        if verbose and ep % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"  [SARSA]       ep {ep:4d}  score {env.score:5d}"
                  f"  avg100 {avg:6.0f}  ε={agent.epsilon:.3f}")

    return scores


def train_dqn(agent: DQNAgent, n_episodes=800, verbose=True):
    """
    Train a DQN agent over n_episodes.

    DQN works with *continuous* states — uses step_continuous() and
    get_continuous_state().  After each step it:
      1. Stores the experience in the replay buffer
      2. Calls train_step() which samples a random batch and updates the network
    """
    env    = DinoEnvironment()
    scores = []

    for ep in range(1, n_episodes + 1):
        env.reset()
        state = env.get_continuous_state()   # DQN always uses continuous state

        while not env.done:
            action     = agent.choose_action(state)
            next_state, reward, done = env.step_continuous(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()     # No-op until buffer has enough samples

            state = next_state

        agent.decay_epsilon()
        scores.append(env.score)

        if verbose and ep % 100 == 0:
            avg = np.mean(scores[-100:])
            buf = len(agent.replay_buffer)
            print(f"  [DQN]         ep {ep:4d}  score {env.score:5d}"
                  f"  avg100 {avg:6.0f}  ε={agent.epsilon:.3f}  buf={buf}")

    return scores


def train_all(n_episodes=800):
    """
    Convenience wrapper — creates fresh agents, trains all three, saves models.
    Returns (q_agent, sarsa_agent, dqn_agent).
    """
    import os
    os.makedirs("models", exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Training all 3 agents  ({n_episodes} episodes each)")
    print(f"{'='*55}\n")

    q     = QLearningAgent()
    sarsa = SARSAAgent()
    dqn   = DQNAgent()

    print("── Q-Learning ──────────────────────────────────")
    train_q_learning(q,    n_episodes)
    q.save()

    print("\n── SARSA ───────────────────────────────────────")
    train_sarsa(sarsa,     n_episodes)
    sarsa.save()

    print("\n── DQN ─────────────────────────────────────────")
    train_dqn(dqn,         n_episodes)
    dqn.save()

    print(f"\n{'='*55}")
    print("  All models saved to models/")
    print(f"{'='*55}\n")

    return q, sarsa, dqn
