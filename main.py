# main.py
"""
Dino Runner — RL Agent Comparison
===================================
Entry point.  Three run modes:

  python main.py                 Train all agents live with Pygame visuals.
                                 (Recommended — watch agents learn in real time!)

  python main.py --no-visual     Headless training (much faster, no window).
                                 After training, run --demo to see results.

  python main.py --demo          Load saved models, watch trained agents play.

  python main.py --episodes 500  Override number of training episodes (default 1000).

Controls while window is open:
  Esc / close window             Stop training / demo.
"""

import argparse
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Dino Runner — RL Agent Comparison (Q-Learning vs SARSA vs DQN)"
    )
    p.add_argument("--no-visual", action="store_true",
                   help="Train without Pygame window (faster)")
    p.add_argument("--demo",      action="store_true",
                   help="Load saved models and watch them play")
    p.add_argument("--episodes",  type=int, default=1000,
                   help="Number of training episodes per agent (default: 1000)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — Visual training  (all 3 agents learn simultaneously on screen)
# ─────────────────────────────────────────────────────────────────────────────

def run_visual_training(n_episodes):
    """
    Main visual training loop.

    Each Pygame frame advances every agent's environment by one step,
    then renders all three panels.  When an episode ends the env is
    reset immediately and the next episode begins.
    """
    import pygame
    from environment import DinoEnvironment
    from agent_q     import QLearningAgent
    from agent_sarsa import SARSAAgent
    from agent_dqn   import DQNAgent
    from game        import init_display, draw_frame, handle_events

    # ── Agents ────────────────────────────────────────────────────────────
    q_agent     = QLearningAgent()
    sarsa_agent = SARSAAgent()
    dqn_agent   = DQNAgent()

    # ── Environments (one per agent) ──────────────────────────────────────
    envs = [DinoEnvironment(), DinoEnvironment(), DinoEnvironment()]

    # ── Per-agent tracking ────────────────────────────────────────────────
    episodes    = [1, 1, 1]
    best_scores = [0, 0, 0]
    ep_done     = [0, 0, 0]     # Episodes completed so far

    # Initial states
    d_states    = [env.reset() for env in envs]       # discrete (Q / SARSA)
    cont_state  = envs[2].get_continuous_state()      # continuous (DQN)

    # SARSA pre-selects its first action (on-policy requirement)
    sarsa_action = sarsa_agent.choose_action(d_states[1])

    # ── Pygame setup ──────────────────────────────────────────────────────
    screen, clock = init_display()
    agents_data   = [
        {"name": "Q-Learning", "env": envs[0]},
        {"name": "SARSA",      "env": envs[1]},
        {"name": "DQN",        "env": envs[2]},
    ]

    print(f"\nVisual training started — {n_episodes} episodes per agent.")
    print("Close the window or press Esc to stop early.\n")

    running = True
    while running and max(ep_done) < n_episodes:

        running = handle_events()

        # ── Q-Learning step ───────────────────────────────────────────────
        if not envs[0].done:
            s  = d_states[0]
            a  = q_agent.choose_action(s)
            ns, r, done = envs[0].step(a)
            q_agent.update(s, a, r, ns, done)
            d_states[0] = ns
            import time
            if done:
                time.sleep(0.3)   # 👈 ADD THIS (pause to show death)
                _ep_end(envs[0], best_scores, ep_done, episodes, 0)
                q_agent.decay_epsilon()
                d_states[0] = envs[0].reset()

        # ── SARSA step ────────────────────────────────────────────────────
        if not envs[1].done:
            s  = d_states[1]
            a  = sarsa_action
            ns, r, done = envs[1].step(a)
            na = sarsa_agent.choose_action(ns)
            sarsa_agent.update(s, a, r, ns, na, done)
            d_states[1]  = ns
            sarsa_action = na
            if done:
                _ep_end(envs[1], best_scores, ep_done, episodes, 1)
                sarsa_agent.decay_epsilon()
                d_states[1]  = envs[1].reset()
                sarsa_action = sarsa_agent.choose_action(d_states[1])

        # ── DQN step ──────────────────────────────────────────────────────
        if not envs[2].done:
            a = dqn_agent.choose_action(cont_state)
            next_cs, r, done = envs[2].step_continuous(a)
            dqn_agent.remember(cont_state, a, r, next_cs, done)
            dqn_agent.train_step()
            cont_state = next_cs
            if done:
                _ep_end(envs[2], best_scores, ep_done, episodes, 2)
                dqn_agent.decay_epsilon()
                envs[2].reset()
                cont_state = envs[2].get_continuous_state()

        # ── Render ────────────────────────────────────────────────────────
        epsilons = [q_agent.epsilon, sarsa_agent.epsilon, dqn_agent.epsilon]
        draw_frame(screen, agents_data, episodes, best_scores, epsilons)
        clock.tick(30)   # Cap at 60 FPS for smooth, human-readable animation

    # ── Save models ───────────────────────────────────────────────────────
    _save_all(q_agent, sarsa_agent, dqn_agent)
    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Headless training  (no Pygame, maximum speed)
# ─────────────────────────────────────────────────────────────────────────────

def run_headless_training(n_episodes):
    """Train all agents without rendering.  Saves models afterwards."""
    from train import train_all
    q, sarsa, dqn = train_all(n_episodes)
    print("Done.  Run `python main.py --demo` to watch the trained agents.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Demo  (load saved models, epsilon = 0, watch them play)
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """Load pre-trained models and display the agents playing."""
    import pygame
    from environment import DinoEnvironment
    from agent_q     import QLearningAgent
    from agent_sarsa import SARSAAgent
    from agent_dqn   import DQNAgent
    from game        import init_display, draw_frame, handle_events

    # Agents with no exploration
    q_agent     = QLearningAgent(epsilon=0.0)
    sarsa_agent = SARSAAgent(epsilon=0.0)
    dqn_agent   = DQNAgent(epsilon=0.0)

    # Load saved weights
    try:
        q_agent.load()
        sarsa_agent.load()
        dqn_agent.load()
    except FileNotFoundError as e:
        print(f"\n  Could not load model: {e}")
        print("  Please train first:  python main.py\n")
        sys.exit(1)

    envs        = [DinoEnvironment(), DinoEnvironment(), DinoEnvironment()]
    d_states    = [env.reset() for env in envs]
    cont_state  = envs[2].get_continuous_state()
    sarsa_action = sarsa_agent.choose_action(d_states[1])

    episodes    = [1, 1, 1]
    best_scores = [0, 0, 0]
    ep_done     = [0, 0, 0]

    agents_data = [
        {"name": "Q-Learning", "env": envs[0]},
        {"name": "SARSA",      "env": envs[1]},
        {"name": "DQN",        "env": envs[2]},
    ]

    screen, clock = init_display("Dino Runner — Demo Mode (Trained Agents)")
    print("\nDemo mode — press Esc or close window to quit.\n")

    running = True
    while running:
        running = handle_events()

        # Q-Learning (greedy — no learning)
        if not envs[0].done:
            s  = d_states[0]
            a  = q_agent.choose_action(s)
            ns, _, done = envs[0].step(a)
            d_states[0] = ns
            if done:
                _ep_end(envs[0], best_scores, ep_done, episodes, 0)
                d_states[0] = envs[0].reset()

        # SARSA (greedy)
        if not envs[1].done:
            s  = d_states[1]
            a  = sarsa_action
            ns, _, done = envs[1].step(a)
            d_states[1]  = ns
            sarsa_action = sarsa_agent.choose_action(ns)
            if done:
                _ep_end(envs[1], best_scores, ep_done, episodes, 1)
                d_states[1]  = envs[1].reset()
                sarsa_action = sarsa_agent.choose_action(d_states[1])

        # DQN (greedy)
        if not envs[2].done:
            a = dqn_agent.choose_action(cont_state)
            next_cs, _, done = envs[2].step_continuous(a)
            cont_state = next_cs
            if done:
                _ep_end(envs[2], best_scores, ep_done, episodes, 2)
                envs[2].reset()
                cont_state = envs[2].get_continuous_state()

        epsilons = [0.0, 0.0, 0.0]
        draw_frame(screen, agents_data, episodes, best_scores, epsilons)
        clock.tick(60)

    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ep_end(env, best_scores, ep_done, episodes, idx):
    """Update tracking counters after an episode ends."""
    if env.score > best_scores[idx]:
        best_scores[idx] = env.score
    ep_done[idx]  += 1
    episodes[idx] += 1


def _save_all(q_agent, sarsa_agent, dqn_agent):
    os.makedirs("models", exist_ok=True)
    q_agent.save()
    sarsa_agent.save()
    dqn_agent.save()
    print("\nAll models saved to models/")
    print("Run `python main.py --demo` to watch the trained agents play.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║         DINO RUNNER  —  RL AGENT COMPARISON          ║
║   Q-Learning  |  SARSA  |  DQN (NumPy)               ║
╚══════════════════════════════════════════════════════╝
""")
    args = parse_args()

    if args.demo:
        run_demo()
    elif args.no_visual:
        run_headless_training(args.episodes)
    else:
        run_visual_training(args.episodes)
