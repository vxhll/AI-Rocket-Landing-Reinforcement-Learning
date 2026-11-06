import os
import gymnasium as gym
from stable_baselines3 import PPO

os.makedirs("videos", exist_ok=True)

models = {
    "original_ppo": "models/ppo_lunar_lander",
    "reward_hacking": "models/ppo_reward_hacking",
    "improved_reward": "models/ppo_improved_reward"
}

for name, model_path in models.items():

    print(f"\nRecording: {name}")

    model = PPO.load(model_path)

    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array"
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=f"videos/{name}",
        episode_trigger=lambda episode: episode == 0,
        name_prefix=name
    )

    obs, info = env.reset()

    total_reward = 0
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

    print(f"Original environment reward: {total_reward:.2f}")

    env.close()

print("\nAll recordings completed!")