# 🦖 Dino Runner — RL Agent Comparison

> A Reinforcement Learning project demonstrating real-time comparison of multiple algorithms using a Dino Runner game environment.

A Chrome Dino-style game where **three reinforcement learning algorithms** learn to dodge cactus obstacles simultaneously, displayed side-by-side in Pygame.

```
┌──────────────┬──────────────┬──────────────┐
│ Q-Learning │ SARSA │ DQN │
│ (blue dino) │(orange dino) │(green dino) │
│ Ep 42 │ Ep 38 │ Ep 51 │
│ Score 820 │ Score 615 │ Score 1100 │
└──────────────┴──────────────┴──────────────┘
```

---

## 🎯 Objective

The goal of this project is to compare the learning behavior of different Reinforcement Learning algorithms in the same dynamic environment. By observing agents side-by-side, the project provides an intuitive understanding of how each algorithm learns and performs.

---

## 🚀 Features

- 🎮 2D Dino Runner game using Pygame  
- 🤖 Multiple RL agents:
  - Q-Learning (Off-policy)
  - SARSA (On-policy)
  - Deep Q-Network (DQN)
- 📊 Real-time side-by-side comparison of agents  
- 📈 Agents learn through trial-and-error using rewards  
- ⚡ Demo mode to visualize trained agents  
- 🔍 Clear visualization of exploration vs exploitation (epsilon decay)

---

## ⚙️ Installation

### Install dependencies
```bash
pip install -r requirements.txt
```
▶️ How to Run
🔹 Train & watch live
python main.py
All three agents train simultaneously in a Pygame window
Watch them fail early, then gradually improve
The epsilon bar shows exploration → exploitation
Press Esc or close the window to stop (models save automatically)

🔹 Headless training (faster)
python main.py --no-visual --episodes 1500
Runs training without UI
Faster execution
Recommended before demo

🔹 Demo — watch trained agents
python main.py --demo
Loads trained models
Runs agents with ε = 0 (no randomness)
Shows learned behavior clearly

🧠 Algorithms
Agent	State type	Algorithm	Key trait
Q-Learning	Discrete	Off-policy TD	Learns optimal greedy policy
SARSA	Discrete	On-policy TD	Learns safer, policy-based actions
DQN	Continuous	Neural Network	Generalizes across states

📊 State Representation
🔹 Discrete (Q-Learning & SARSA)
state = distance_bucket (0–10) × 2 + is_jumping (0–1)
🔹 Continuous (DQN)
[dist_to_cactus, cactus_height, dino_airtime, vertical_velocity, speed]
🎯 Reward Function
+1   for every frame survived
-100 on collision (episode ends)

This reward structure encourages the agent to survive longer and avoid obstacles.

🗂️ Project Structure
dino_rl/
├── main.py          Entry point (train / demo / headless)
├── environment.py   Game logic, obstacles, collision, state encoding
├── agent_q.py       Q-Learning (Q-table, epsilon-greedy)
├── agent_sarsa.py   SARSA (on-policy, Q-table)
├── agent_dqn.py     DQN (NumPy neural network, experience replay)
├── game.py          Pygame renderer — three side-by-side panels
├── train.py         Training loops for all agents
├── models/          Saved trained models
└── requirements.txt

🎮 Controls
Key / Action	Effect
Esc	Stop and save models
Window close (×)	Stop and save models

📌 Key Observations
Q-Learning learns quickly but may take aggressive actions
SARSA is more stable and cautious
DQN performs better in dynamic environments due to generalization

👩‍💻 Author
Anjali Pillai
