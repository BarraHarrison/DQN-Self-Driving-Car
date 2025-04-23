# DQN Self-Driving Car

## Introduction
Welcome to the **DQN Self-Driving Car** simulation project! This project is a reinforcement learning-based simulation where a self-driving car learns to navigate a race track using a Deep Q-Network (DQN). The simulation is built with Python and Pygame and allows users to train, evaluate, and monitor the performance of an autonomous agent in real-time. The goal is to explore how AI can learn driving behavior through rewards and trial-and-error.

---

## Features

### 🚗 Self-driving car agent with sensor-based perception
The car is equipped with simulated sensors that detect distances to nearby obstacles in five directions. These sensor readings form the state input for the agent, helping it make navigation decisions such as when to turn or go straight.

### 🧠 Deep Q-Network (DQN) with experience replay
The agent learns using a DQN, which approximates the optimal action-value function. Experience replay allows the agent to store and sample past experiences (state, action, reward, next state, done), improving learning stability and efficiency.

### 🧪 Evaluation mode using a trained model (.pth)
Once a model is trained, it can be saved as a `.pth` file. In evaluation mode, the agent runs using this trained model with a fixed epsilon value, allowing users to test performance without additional learning.

### 📊 Real-time episode reward visualization
The simulation displays a live-updating graph that plots total rewards obtained per episode, helping users monitor the agent's progress and performance over time.

### 🏁 Lap tracking and timing
The simulation tracks laps completed by the car and calculates the time taken for each lap. This feature helps assess how efficiently the agent can complete a full circuit.

### 📈 Session summary (average reward, best lap time, total laps)
At the end of the session, a detailed summary is printed in the terminal. It includes the average episode reward, the best recorded lap time, and the total number of laps completed during the session.

### 🖼️ Custom race track and car sprite support
The simulation supports loading custom track images (`.png`) and car sprites, allowing for experimentation with different maps and car designs.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/BarraHarrison/DQN-Self-Driving-Car.git
cd DQN-Self-Driving-Car

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt

# Run the main simulation
python3 main.py
```

To switch to evaluation mode, set `EVAL_ONLY = True` in `main.py` and ensure a model file exists in the `checkpoints/` folder.

---

## Conclusion
This project showcases how reinforcement learning can be applied to a simplified self-driving simulation. Through real-time feedback, reward tracking, and lap timing, users can observe how an AI agent gradually learns efficient driving strategies. It serves as both an educational tool and a stepping stone toward more advanced autonomous driving experiments.

Happy training! 🧠🚗

