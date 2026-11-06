
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Load trained PPO model
model = PPO.load("models/ppo_lunar_lander")

# Create Lunar Lander environment
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

# Evaluation settings
episodes = 10
rewards = []
steps_list = []
successful_landings = 0

print("\nStarting AI Evaluation...\n")

for episode in range(episodes):

    observation, info = env.reset()

    total_reward = 0
    steps = 0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # Predict action using trained model
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(
            int(action)
        )

        total_reward += reward
        steps += 1

    rewards.append(total_reward)
    steps_list.append(steps)

    if total_reward >= 200:
        successful_landings += 1

    print(f"Episode: {episode + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {steps}")
    print("-" * 35)

# Calculate evaluation metrics
average_reward = np.mean(rewards)
average_steps = np.mean(steps_list)

print("\n========== EVALUATION RESULTS ==========")
print(f"Total Episodes: {episodes}")
print(f"Average Reward: {average_reward:.2f}")
print(f"Average Steps: {average_steps:.2f}")
print(f"Successful Episodes (Reward >= 200): {successful_landings}/{episodes}")
print("========================================")

env.close()
