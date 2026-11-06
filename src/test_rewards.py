
import gymnasium as gym

from reward_shaping import (
    TimePenaltyWrapper,
    RewardHackingWrapper,
    ImprovedRewardWrapper
)

experiments = {
    "Time Penalty": TimePenaltyWrapper,
    "Reward Hacking": RewardHackingWrapper,
    "Improved Reward": ImprovedRewardWrapper
}

for name, wrapper in experiments.items():

    env = gym.make("LunarLander-v3")
    env = wrapper(env)

    observation, info = env.reset()

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(f"{name}:")
    print(f"Reward: {reward:.4f}")
    print("-" * 30)

    env.close()
