
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Load reward hacking model
model = PPO.load("models/ppo_reward_hacking")

# Use original environment without reward wrapper
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

episodes = 10
rewards = []
steps_list = []

print("\nEvaluating Reward Hacking Agent...\n")

for episode in range(episodes):

    obs, info = env.reset()

    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            int(action)
        )

        total_reward += reward
        steps += 1

    rewards.append(total_reward)
    steps_list.append(steps)

    print(f"Episode: {episode + 1}")
    print(f"Original Environment Reward: {total_reward:.2f}")
    print(f"Steps: {steps}")
    print("-" * 35)

print("\n========== RESULTS ==========")
print("Average Original Reward:", round(np.mean(rewards), 2))
print("Average Episode Length:", round(np.mean(steps_list), 2))
print("=============================")

env.close()