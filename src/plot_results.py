import matplotlib.pyplot as plt
import numpy as np

models = [
    "Original PPO",
    "Reward Hacking",
    "Improved Reward"
]

rewards = [151.49, 2.70, 27.48]

plt.figure(figsize=(10, 6))

bars = plt.bar(models, rewards)

plt.axhline(
    y=200,
    linestyle="--",
    label="Solved threshold (200)"
)

plt.title("AI Rocket Landing - Model Comparison")
plt.xlabel("Reinforcement Learning Model")
plt.ylabel("Average Original Environment Reward")

plt.legend()
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, rewards):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.show()
