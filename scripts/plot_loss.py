import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
loss_path = PROJECT_ROOT / "outputs" / "loss_history.json"

with open(loss_path, "r") as f:
    data = json.load(f)

steps = data["step"]
train_loss = data["train_loss"]
val_loss = data["val_loss"]

plt.figure()
plt.plot(steps, train_loss, label="train")
plt.plot(steps, val_loss, label="val")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.legend()

save_path = PROJECT_ROOT / "outputs" / "loss_curve.png"
plt.savefig(save_path)

print(f"Saved plot to: {save_path}")
plt.show()