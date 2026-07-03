# rl-implementations

Small RL and imitation-learning implementations. The current Robomimic work lives in `architectures/flow_policy`:

- `train.py`: offline BC training for the flow policy
- `eval_rollout.py`: Robosuite rollout eval and video recording
- `ogpo.py`: online OGPO-style training from a BC flow-policy checkpoint

## Setup

Use a virtualenv from the repo root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On an NVIDIA GPU machine, install the CUDA PyTorch wheel first. Pick the CUDA wheel that matches the machine's driver; this is a common CUDA 12.4 example:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

On Mac or CPU-only Linux, this is usually enough:

```bash
python -m pip install -r requirements.txt
```

Check that PyTorch sees the device you expect:

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("mps:", torch.backends.mps.is_available())
PY
```

For headless Robosuite rendering on a Linux GPU machine, these env vars are often needed:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If Robosuite or Robomimic prints macro warnings, they are usually harmless. To create local macro files:

```bash
python -m robosuite.scripts.setup_macros
python -m robomimic.scripts.setup_macros
```

## Robomimic Data

Download Lift low-dimensional data:

```bash
mkdir -p datasets/robomimic
python -m robomimic.scripts.download_datasets \
  --download_dir datasets/robomimic \
  --tasks lift \
  --dataset_types ph \
  --hdf5_types low_dim
```

The Lift path used by the code is:

```text
datasets/robomimic/lift/ph/low_dim_v15.hdf5
```

For Square multi-human data:

```bash
python -m robomimic.scripts.download_datasets \
  --download_dir datasets/robomimic \
  --tasks square \
  --dataset_types mh \
  --hdf5_types low_dim
```

Expected Square path:

```text
datasets/robomimic/square/mh/low_dim_v15.hdf5
```

## Train Flow BC

Lift:

```bash
python -m architectures.flow_policy.train \
  --dataset datasets/robomimic/lift/ph/low_dim_v15.hdf5 \
  --output outputs/flow_policy_lift.pt \
  --total-steps 20000 \
  --device cuda
```

Use `--device auto` if you want the code to choose CUDA, MPS, then CPU.

Square:

```bash
python -m architectures.flow_policy.train \
  --dataset datasets/robomimic/square/mh/low_dim_v15.hdf5 \
  --output outputs/flow_policy_square_mh.pt \
  --total-steps 50000 \
  --device cuda
```

## Rollout Eval

Lift eval:

```bash
python -m architectures.flow_policy.eval_rollout \
  --checkpoint outputs/flow_policy_lift.pt \
  --dataset datasets/robomimic/lift/ph/low_dim_v15.hdf5 \
  --episodes 20 \
  --device cuda
```

Record videos:

```bash
python -m architectures.flow_policy.eval_rollout \
  --checkpoint outputs/flow_policy_lift.pt \
  --dataset datasets/robomimic/lift/ph/low_dim_v15.hdf5 \
  --episodes 5 \
  --video-dir outputs/flow_policy_lift_videos \
  --device cuda
```

If videos are upside down on a new machine, add `--no-flip-video`.

## Train OGPO

Start from a BC flow checkpoint. W&B logging is optional.

```bash
python -m architectures.flow_policy.ogpo \
  --policy-checkpoint outputs/flow_policy_lift.pt \
  --dataset datasets/robomimic/lift/ph/low_dim_v15.hdf5 \
  --output outputs/ogpo_lift.pt \
  --device cuda \
  --wandb \
  --wandb-project robomimic-flow-policy \
  --wandb-name ogpo-lift
```

Without W&B, remove the `--wandb ...` args.

For a quick smoke test:

```bash
python -m architectures.flow_policy.ogpo \
  --policy-checkpoint outputs/flow_policy_lift.pt \
  --dataset datasets/robomimic/lift/ph/low_dim_v15.hdf5 \
  --output outputs/ogpo_lift_smoke.pt \
  --total-env-steps 64 \
  --start-transitions 16 \
  --batch-size 16 \
  --sample-steps 4 \
  --actor-samples 2 \
  --actor-update-epochs 1 \
  --num-qs 4 \
  --log-every 16 \
  --save-every 64 \
  --device cuda
```

## Notes

- A CUDA GPU is strongly preferred for OGPO updates. Mac MPS can run small tests, but the sampled flow chains and REDQ ensemble are much slower.
- Robosuite simulation still has CPU overhead, so online training speed will not scale like pure supervised BC training.
- `outputs/`, `datasets/`, and local `wandb/` logs are gitignored.
- The ACT robot scripts under `architectures/act` and `scripts` may need extra hardware-specific dependencies such as `lerobot`. They are not required for the Robomimic flow-policy path.
