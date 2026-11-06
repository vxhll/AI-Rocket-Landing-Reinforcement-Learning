# 🚀 AI Rocket Landing | Reinforcement Learning

### Training an autonomous lunar lander using PPO, reward shaping & reward hacking experiments.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Algorithm-PPO-00A67E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gymnasium-LunarLander-blue?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Reinforcement%20Learning-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/Experiments-3-16A34A?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=flat-square" />
</p>

---

## 🌌 Overview

Can an AI learn to land a rocket without being explicitly programmed how to fly?

This project explores autonomous rocket landing using **Reinforcement Learning (RL)**. An AI agent interacts with the Lunar Lander simulation, learns through trial and error, and improves its decision-making using rewards and penalties.

Beyond training an autonomous agent, this project investigates a critical challenge in AI: **Reward Hacking** — when an agent discovers unintended ways to maximize its reward instead of achieving the actual objective.

### 🎯 Project Highlights

- Autonomous rocket landing using PPO.
- Random agent baseline implementation.
- Neural network policy learning with PyTorch.
- Custom reward shaping experiments.
- Reward hacking demonstration.
- Evaluation across multiple episodes.
- Simulation recordings and performance visualization.

---

## 🛰️ Environment

**Gymnasium – LunarLander-v3**

The agent interacts with a physics-based environment and receives eight numerical observations at each timestep.

| Observation | Description |
|:---|:---|
| X Position | Horizontal position |
| Y Position | Vertical position |
| X Velocity | Horizontal movement |
| Y Velocity | Vertical movement |
| Angle | Rocket orientation |
| Angular Velocity | Rotation speed |
| Left Leg Contact | Left landing leg status |
| Right Leg Contact | Right landing leg status |

### 🎮 Action Space

| Action | Control |
|:---:|:---|
| 0 | Do Nothing |
| 1 | Fire Left Engine |
| 2 | Fire Main Engine |
| 3 | Fire Right Engine |

---

## 🧠 Model Architecture

The project uses **Proximal Policy Optimization (PPO)**, a policy-gradient reinforcement learning algorithm.

**Neural Network Architecture**

```text
        Environment
             |
             ▼
    8 Numerical Observations
             |
             ▼
      Input Layer (8)
             |
             ▼
     Hidden Layer (64)
             |
             ▼
     Hidden Layer (64)
             |
       ┌─────┴─────┐
       ▼           ▼
  Policy Network  Value Network
       |           |
       ▼           ▼
  4 Actions    State Value
       |
       ▼
   Rocket Control
```

### ⚙️ Training Configuration

| Hyperparameter | Value |
|:---|---:|
| Algorithm | PPO |
| Policy | MlpPolicy |
| Learning Rate | 0.0003 |
| Batch Size | 64 |
| Rollout Steps | 2048 |
| Epochs | 10 |
| Discount Factor | 0.99 |
| GAE Lambda | 0.95 |
| Network | [64, 64] |

---

## 🔬 Experiments

Three independent PPO models were trained to investigate the impact of different reward functions.

### 01 — Original PPO

The baseline agent learns using the built-in Lunar Lander reward function.

- Training timesteps: 1,000,000
- Reward function: Environment default
- Objective: Achieve safe autonomous landing.

### 02 — Reward Hacking

A custom reward bonus was introduced:

```python
if (
    abs(angle) < 0.1
    and abs(x_velocity) < 0.1
    and abs(y_velocity) < 0.1
):
    reward += 1.0
```

The intention was to encourage stable rocket behaviour.

However, this reward can encourage the agent to remain in a stable state for longer rather than complete the landing task.

- Training timesteps: 500,000
- Additional reward: +1 for satisfying stability conditions
- Objective: Investigate unintended reward optimization.

### 03 — Improved Reward Shaping

A stability-improvement reward was introduced:

```python
improvement = previous_stability - stability

reward += 0.5 * improvement
reward -= 0.01
```

This experiment rewards improvements in stability and introduces a small time penalty.

- Training timesteps: 500,000
- Reward: Stability improvement
- Additional penalty: -0.01 per frame
- Objective: Investigate whether progress-based rewards improve landing behaviour.

---

## 📊 Performance Dashboard

All models were evaluated using the original Lunar Lander environment, without their custom reward wrappers.

### Model Comparison

| Model | Average Reward | Average Steps | Success Rate |
|:---|---:|---:|---:|
| Original PPO | **151.49** | 443.10 | 60% |
| Reward Hacking PPO | 2.70 | 980.50 | Not recorded |
| Improved Reward PPO | 27.48 | 396.10 | 20% |

<p align="center">
  <img src="results/model_comparison.png" alt="PPO Model Comparison" width="90%" />
</p>

<details>
<summary><b>📈 Detailed Experimental Findings</b></summary>

**Original PPO**
- Average reward: 151.49
- Successful episodes: 6/10
- Best-performing model across the reported evaluations.

**Reward Hacking PPO**
- Average original reward: 2.70
- Nine out of ten episodes reached the 1,000-step time limit.
- Demonstrated behaviour consistent with prolonged reward-seeking rather than reliable task completion.

**Improved Reward PPO**
- Average original reward: 27.48
- Successful episodes: 2/10
- The experimental stability-based reward did not outperform the original PPO baseline.

Success is defined as an original environment episode reward of at least 200.

</details>

---

## 🎥 Simulation Gallery

Three autonomous agent demonstrations were recorded using Gymnasium's `RecordVideo` wrapper.

| Experiment | Recording |
|:---|:---|
| Original PPO | [▶ View Simulation](videos/original_ppo/) |
| Reward Hacking PPO | [▶ View Simulation](videos/reward_hacking/) |
| Improved Reward PPO | [▶ View Simulation](videos/improved_reward/) |

*Open the respective folders to access the MP4 recordings.*

---

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Gymnasium-000000?style=flat-square" />
  <img src="https://img.shields.io/badge/Stable--Baselines3-2C3E50?style=flat-square" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" />
  <img src="https://img.shields.io/badge/TensorBoard-FF6F00?style=flat-square" />
</p>

---

## 📁 Repository Structure

```text
AI-Rocket-Landing-RL/
│
├── environment/
│   └── test_environment.py
│
├── src/
│   ├── random_agent.py
│   ├── train_ppo.py
│   ├── evaluate_agent.py
│   ├── reward_shaping.py
│   ├── test_rewards.py
│   ├── train_reward_hacking.py
│   ├── evaluate_reward_hacking.py
│   ├── train_improved_reward.py
│   ├── evaluate_improved_reward.py
│   ├── plot_results.py
│   └── record_videos.py
│
├── models/
├── videos/
│   ├── original_ppo/
│   ├── reward_hacking/
│   └── improved_reward/
│
├── results/
│   ├── experiment_summary.csv
│   └── model_comparison.png
│
├── logs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vxhll/AI-Rocket-Landing-RL.git

cd AI-Rocket-Landing-RL
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the original PPO agent

```bash
python src/train_ppo.py
```

### 6. Evaluate the trained agent

```bash
python src/evaluate_agent.py
```

### 7. Run reward experiments

```bash
python src/train_reward_hacking.py

python src/train_improved_reward.py
```

### 8. Generate visualizations

```bash
python src/plot_results.py
```

---

## 🔭 Future Improvements

- Implement potential-based reward shaping.
- Experiment with different reward coefficients.
- Tune PPO hyperparameters.
- Evaluate multiple random seeds.
- Compare PPO with A2C and SAC where appropriate.
- Improve autonomous landing success rate.
- Investigate reward robustness and unintended agent behaviour.

---

## 💡 Key Takeaway

> An AI agent optimizes the reward function we provide, not necessarily the objective we intend.

This project demonstrates why reward design, behavioural evaluation, and validation against the original task are essential in reinforcement learning.

Understanding why an agent fails is as important as understanding why it succeeds.

---

## 👨‍💻 Author

**Vishal J**

AI & Machine Learning Engineer

<p>
  <a href="https://github.com/vxhll">
    <img src="https://img.shields.io/badge/GitHub-vxhll-181717?style=for-the-badge&logo=github" />
  </a>
  <a href="https://www.linkedin.com/in/vishal-j-btech">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin" />
  </a>
</p>

---

<p align="center">
  <b>Built with Python, Reinforcement Learning & curiosity.</b>
  <br />
  ⭐ If you find this project interesting, consider starring the repository.
</p>
