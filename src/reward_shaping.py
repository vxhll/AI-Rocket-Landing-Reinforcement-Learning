
import gymnasium as gym
import numpy as np


class TimePenaltyWrapper(gym.Wrapper):
    """Experiment A: Penalize every extra frame."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward -= 0.01

        return obs, reward, terminated, truncated, info


class RewardHackingWrapper(gym.Wrapper):
    """Experiment B: Reward hovering while upright and slow."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_velocity = obs[2]
        y_velocity = obs[3]
        angle = obs[4]

        if (
            abs(angle) < 0.1
            and abs(x_velocity) < 0.1
            and abs(y_velocity) < 0.1
        ):
            reward += 1.0

        return obs, reward, terminated, truncated, info


class ImprovedRewardWrapper(gym.Wrapper):
    """Experiment C: Reward improvement in stability."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_stability = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_velocity = obs[2]
        y_velocity = obs[3]
        angle = obs[4]

        stability = (
            abs(angle)
            + abs(x_velocity)
            + abs(y_velocity)
        )

        improvement = self.previous_stability - stability

        reward += 0.5 * improvement
        reward -= 0.01

        self.previous_stability = stability

        if terminated or truncated:
            self.previous_stability = 0.0

        return obs, reward, terminated, truncated, info
