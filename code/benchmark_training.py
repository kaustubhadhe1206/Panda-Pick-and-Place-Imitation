"""Measure how fast Diffusion Policy training runs on this GPU with different speed settings.

Uses random data of the real shapes (no simulator needed), so it takes ~2 minutes.
Each row adds one setting on top of the previous one:
  plain             fp32, cuDNN picks algorithms without testing, PyTorch's default AdamW,
                    EMA one tensor at a time (how training ran before the speed fixes)
  + cudnn benchmark cuDNN tries its convolution algorithms once and keeps the fastest
  + fused + foreach fused AdamW kernel and batched EMA update (what train.py now does by default)
  + fp16 (use_amp)  mixed precision on top of that (train.use_amp=true)
Only the last one changes the numbers the network computes with; the others are pure speed-ups.

Usage:
  python code/benchmark_training.py [--steps 100]
"""

import argparse
import json
import time

import torch

import hw2  # noqa: F401
from hw2 import utils
from hw2.policy import DiffusionPolicy, ExponentialMovingAverage

SETTINGS = [
    ("plain", dict(cudnn_benchmark=False, fused_and_foreach=False, use_amp=False)),
    ("+ cudnn benchmark", dict(cudnn_benchmark=True, fused_and_foreach=False, use_amp=False)),
    ("+ fused + foreach", dict(cudnn_benchmark=True, fused_and_foreach=True, use_amp=False)),
    ("+ fp16 (use_amp)", dict(cudnn_benchmark=True, fused_and_foreach=True, use_amp=True)),
]


def ema_update_one_tensor_at_a_time(ema, model):
  """The old (slow) EMA update, kept only for this comparison."""
  decay = min(ema.max_decay, 1.0 - (1.0 + ema.num_updates) ** -ema.power)
  ema.num_updates += 1
  with torch.no_grad():
    for averaged_param, param in zip(ema.averaged_model.parameters(), model.parameters()):
      averaged_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def measure_steps_per_second(config, device, num_steps, cudnn_benchmark, fused_and_foreach, use_amp):
  torch.backends.cudnn.benchmark = cudnn_benchmark
  policy_config, train_config = config["policy"], config["train"]
  torch.manual_seed(0)
  obs_dim, action_dim = 66, 8
  policy = DiffusionPolicy(obs_dim=obs_dim, action_dim=action_dim, **policy_config).to(device)
  ema = ExponentialMovingAverage(policy, power=train_config["ema_power"])
  optimizer = torch.optim.AdamW(policy.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"],
                                fused=fused_and_foreach)   # not fused = PyTorch's default AdamW
  grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
  batch_size = train_config["batch_size"]
  fake_obs = torch.rand(batch_size, policy_config["obs_horizon"], obs_dim, device=device) * 2 - 1
  fake_actions = torch.rand(batch_size, policy_config["pred_horizon"], action_dim, device=device) * 2 - 1

  def one_step():
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
      loss = policy.compute_loss(fake_obs, fake_actions)
    optimizer.zero_grad(set_to_none=True)
    grad_scaler.scale(loss).backward()
    grad_scaler.step(optimizer)
    grad_scaler.update()
    if fused_and_foreach:
      ema.update(policy)
    else:
      ema_update_one_tensor_at_a_time(ema, policy)

  for _ in range(10):   # warm up (and let cuDNN pick its algorithms)
    one_step()
  torch.cuda.synchronize()
  start_time = time.time()
  for _ in range(num_steps):
    one_step()
  torch.cuda.synchronize()
  return num_steps / (time.time() - start_time)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  utils.add_config_arguments(parser, "configs/dp_base.yaml")
  parser.add_argument("--steps", type=int, default=100, help="timed training steps per setting")
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise SystemExit("This benchmark needs a GPU.")
  config = utils.load_config(args.config, args.overrides)
  device = torch.device("cuda")
  num_train_steps = config["train"]["num_steps"]

  results = []
  for name, settings in SETTINGS:
    steps_per_second = measure_steps_per_second(config, device, args.steps, **settings)
    results.append({"setting": name, "steps_per_second": round(steps_per_second, 2),
                    "minutes_per_run": round(num_train_steps / steps_per_second / 60, 1)})
    print(f"{name:20s} {steps_per_second:6.2f} steps/s -> {results[-1]['minutes_per_run']:6.1f} min "
          f"for {num_train_steps} steps", flush=True)

  print("\n==== TRAINING SPEED (paste this back) ====")
  print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "batch_size": config["train"]["batch_size"], "num_steps": num_train_steps,
                    "results": results}, indent=1))


if __name__ == "__main__":
  main()
