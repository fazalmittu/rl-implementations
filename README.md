# rl-implementations

Minimal, readable implementations of important RL algorithms. The goal is to
understand the **core logic** of each method — not to be a fast or general
framework. Each algorithm lives in one file with its math heavily annotated;
reusable infrastructure (replay buffer, networks, logging) is shared in
`common/`.

**Rule of thumb: infrastructure is shared, the math is local.**

## Layout

```
common/          # shared infrastructure (do not put algorithm math here)
  replay_buffer.py   circular off-policy buffer -> sample() torch minibatches
  networks.py        MLP factory
  logger.py          scalar logging + learning-curve PNG
algorithms/      # one file per algorithm, self-contained core logic
  td3.py
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m algorithms.td3      # trains on Pendulum-v1, saves a learning curve
```

## Algorithms

| # | Algorithm | One key idea | Status |
|---|-----------|--------------|--------|
| 1 | **TD3** | Twin clipped critics + delayed actor + target smoothing fix Q-overestimation | ✅ |
| 2 | SAC | Maximum-entropy objective for exploration + sample efficiency | ⬜ |
| 3 | HER | Relabel failed goals as achieved ones for sparse-reward robot tasks | ⬜ |
| 4 | REINFORCE | The score-function (log-prob) policy gradient | ⬜ |
| 5 | A2C | Advantage actor-critic | ⬜ |
| 6 | GAE | Bias/variance-tunable advantage estimation (used by TRPO/PPO) | ⬜ |
| 7 | TRPO | Trust-region constraint on the policy update | ⬜ |
| 8 | PPO | TRPO's trust region as a cheap clipped objective | ⬜ |
| 9 | BC | Supervised imitation — the baseline for diffusion/flow policies | ⬜ |
| 10 | Diffusion Policy | Model the action distribution as a denoising process | ⬜ |
| 11 | Flow Matching Policy | Learn a velocity field mapping noise → actions | ⬜ |
| 12 | IQL / CQL | Offline RL without querying out-of-distribution actions | ⬜ |
| 13 | Decision Transformer | RL as return-conditioned sequence modeling | ⬜ |
| 14 | Dreamer | Learn a world model, train the policy in imagination | ⬜ |

Built in sequence; each algorithm reuses `common/` where it can.
