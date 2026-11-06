
import gymnasium as gym
import numpy as np

# Create Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset environment
observation, info = env.reset()

print("\nInitial Observation:")
print(observation)

print("\nObservation Shape:")
print(observation.shape)

print("\nAction Space:")
print(env.action_space)

print("\nAction Meanings:")
actions = {
    0: "Do Nothing",
    1: "Fire Left Engine",
    2: "Fire Main Engine",
    3: "Fire Right Engine"
}

for action, meaning in actions.items():
    print(f"{action}: {meaning}")

# Perform one random action
action = env.action_space.sample()

observation, reward, terminated, truncated, info = env.step(action)

print("\nAfter One Action:")
print("Action:", action)
print("Observation:", observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()