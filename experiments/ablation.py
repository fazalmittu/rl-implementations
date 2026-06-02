"""Ablation harness: run normal TD3 vs. one modification on the SAME seed and
overlay the results, so you can see what a given piece of TD3 actually buys you.

Two panels are plotted:
  - episode return        (does it still learn / how well?)
  - mean critic Q estimate (does the value estimate stay sane or blow up?)

The Q panel is the interesting one for these ablations: removing the target
networks tends to make the critic's own estimate diverge long before the
return curve reacts.

Usage:
    python -m experiments.ablation target_actor    # use LIVE actor in the target
    python -m experiments.ablation target_critic   # use LIVE critic in the target
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms.td3 import train

# name -> (human label, kwargs that turn the piece OFF)
ABLATIONS = {
    "target_actor":  ("live actor in target",  dict(use_target_actor=False)),
    "target_critic": ("live critic in target", dict(use_target_critic=False)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ablation", choices=list(ABLATIONS))
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    label, kwargs = ABLATIONS[args.ablation]

    print(f"=== baseline: normal TD3 on {args.env} ===")
    base = train(env_id=args.env, total_steps=args.steps, seed=args.seed, label="TD3 (baseline)")
    print(f"=== ablation: {label} on {args.env} ===")
    abl = train(env_id=args.env, total_steps=args.steps, seed=args.seed, label=label, **kwargs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(base["steps"], base["returns"], "-o", label="TD3 (baseline)")
    ax1.plot(abl["steps"], abl["returns"], "-o", label=label)
    ax1.set(xlabel="env steps", ylabel="episode return", title="Return (higher = better)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(base["steps"], base["q_values"], "-o", label="TD3 (baseline)")
    ax2.plot(abl["steps"], abl["q_values"], "-o", label=label)
    ax2.set(xlabel="env steps", ylabel="mean critic Q", title="Critic's own Q estimate")
    ax2.legend(); ax2.grid(alpha=0.3)

    out = f"experiments/ablation_{args.ablation}_{args.env}.png"
    fig.suptitle(f"TD3 vs. {label}  ({args.env})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved comparison -> {out}")


if __name__ == "__main__":
    main()
