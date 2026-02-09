# ♟️ Chess Agent Bot — Deep Reinforcement Learning

An intelligent **chess-playing AI** built using **Deep Q-Learning (DQL)** that learns to play chess through **self-play and reinforcement learning**.  
The agent improves over time by evaluating board states, generating legal moves, and optimizing long-term rewards.

---

## 🚀 Project Overview

This project focuses on building a **reinforcement learning–based chess engine** rather than relying on classical minimax or hard-coded heuristics.

The agent:
- Learns from **self-play**
- Understands **complete chess rules**
- Improves decision-making through **reward optimization**
- Interacts with a **Pygame-based chess environment**

---

## 🧠 Key Features

- ♞ Deep Q-Learning (DQL) for move selection  
- ♜ Complete chess rule encoding (legal moves, captures, checks, etc.)
- 🔁 Self-play training environment
- 🎯 Custom reward function for strategic learning
- 📈 Performance improvement after training
- 🎮 Pygame-based GUI for visualization

---

## 🧩 Tech Stack

| Category | Tools |
|--------|------|
| Programming Language | Python |
| Machine Learning | Deep Q-Learning (DQL) |
| Libraries | NumPy, TensorFlow |
| Game Engine | Pygame |
| Environment | Custom Chess Environment |
| Version Control | Git & GitHub |

---

## 🎯 How the Agent Learns

1. Observes the current board state  
2. Generates all legal moves  
3. Selects an action using ε-greedy policy  
4. Receives a reward  
5. Updates Q-values using the Bellman Equation  
6. Improves strategy over thousands of games  

---

## 🏆 Reward Strategy (Simplified)

| Action | Reward |
|------|-------|
| Capture opponent piece | Positive |
| Checkmate | High Positive |
| Illegal move | Negative |
| Losing piece | Negative |
| Winning game | High Positive |

---

## 👤 Author
- pardhu01010
