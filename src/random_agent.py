
import gymnasium as gym

# Create environment
env = gym.make("LunarLander-v3", render_mode="human")

episodes = 5

for episode in range(episodes):

    observation, info = env.reset()

    total_reward = 0
    step_count = 0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # Select a random action
        action = env.action_space.sample()

        # Perform action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

    print(f"\nEpisode: {episode + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {step_count}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

env.close()
