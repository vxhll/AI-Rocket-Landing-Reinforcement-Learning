
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from reward_shaping import RewardHackingWrapper
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create Lunar Lander environment
env = gym.make("LunarLander-v3")

# Apply reward hacking wrapper
env = RewardHackingWrapper(env)
env = Monitor(env)

# Initialize PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    ),
    verbose=1,
    tensorboard_log="./logs/"
)

print("\nTraining Reward Hacking Agent...\n")

# Train the agent
model.learn(
    total_timesteps=500_000,
    progress_bar=True,
    tb_log_name="Reward_Hacking"
)

# Save separately
model.save("models/ppo_reward_hacking")

print("\nReward Hacking Training Completed!")

env.close()