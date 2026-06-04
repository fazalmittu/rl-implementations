"""Watch a trained policy act in the environment.

Records a GIF of a few greedy episodes (rgb_array frames -> imageio), so you can
replay what the policy actually learned. Pass --human for a live pygame window
instead.

Usage:
    python -m experiments.watch                 # train REINFORCE, save a gif
    python -m experiments.watch --human         # live window instead of a gif
"""

import argparse

import gymnasium as gym
import imageio
import torch

from algorithms.reinforce import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--human", action="store_true", help="live window instead of a gif")
    args = p.parse_args()

    # train an agent, then watch its greedy policy
    print("=== training ===")
    agent = train(env_id=args.env)["agent"]

    print("=== rolling out greedy policy ===")
    render_mode = "human" if args.human else "rgb_array"
    env = gym.make(args.env, render_mode=render_mode)
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            if not args.human:
                frames.append(env.render())
            with torch.no_grad():
                action = agent.policy(torch.as_tensor(obs, dtype=torch.float32)).argmax().item()
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += reward
        print(f"episode {ep + 1}: return = {ep_ret:.0f}")
    env.close()

    if not args.human:
        out = f"experiments/watch_{args.env}.gif"
        imageio.mimsave(out, frames, fps=30)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
