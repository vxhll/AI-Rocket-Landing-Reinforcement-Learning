
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Load improved reward model
model = PPO.load("models/ppo_improved_reward")

# Original environment (no custom reward wrapper)
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

episodes = 10
rewards = []
steps_list = []
successful_episodes = 0

print("\nEvaluating Improved Reward Agent...\n")

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

    if total_reward >= 200:
        successful_episodes += 1

    print(f"Episode: {episode + 1}")
    print(f"Original Reward: {total_reward:.2f}")
    print(f"Steps: {steps}")
    print("-" * 35)

print("\n========== EVALUATION RESULTS ==========")
print(f"Average Reward: {np.mean(rewards):.2f}")
print(f"Average Steps: {np.mean(steps_list):.2f}")
print(f"Successful Episodes: {successful_episodes}/{episodes}")
print("========================================")

env.close()