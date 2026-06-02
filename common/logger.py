"""Minimal scalar logging + learning-curve plotting.

Every algorithm should `log(step, return)` during training and `plot()` at the
end, so you can *see* learning happen. Seeing the curve go up is the best
sanity check that an implementation is correct.
"""

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, name):
        self.name = name
        self.steps = []
        self.values = []

    def log(self, step, value):
        self.steps.append(step)
        self.values.append(value)
        print(f"[{self.name}] step={step:>8}  return={value:8.2f}")

    def plot(self, path):
        plt.figure()
        plt.plot(self.steps, self.values)
        plt.xlabel("environment steps")
        plt.ylabel("episode return")
        plt.title(self.name)
        plt.grid(True, alpha=0.3)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"[{self.name}] saved learning curve -> {path}")
