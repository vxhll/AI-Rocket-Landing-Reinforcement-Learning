
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create and monitor environment
env = gym.make("LunarLander-v3")
env = Monitor(env)

# Configure PPO
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        )
    ),
    verbose=1,
    tensorboard_log="./logs/"
)

print("\nStarting PPO Training...\n")

# Train the agent
model.learn(
    total_timesteps=1_000_000,
    progress_bar=True
)

# Save trained model
model.save("models/ppo_lunar_lander")

print("\nTraining completed!")
print("Model saved successfully!")

env.close()
