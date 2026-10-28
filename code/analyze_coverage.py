"""Does a student fail more often on layouts that are far from its training demos? (Part 4 failure analysis)

Every episode's layout (pick spot + place spot) follows from its seed alone, so we
recompute the layouts of the demo seeds and the evaluation seeds with the same code
the env uses (no simulation, runs on CPU in a few seconds). For each evaluation
layout we measure the distance to the closest layout in the training subset:
  nearest-demo distance = min over training demos of || (pick_xy, place_xy) - (demo pick_xy, demo place_xy) ||
and compare it with the student's success rate on that layout (averaged over the 3 training seeds).

Writes:
  results/coverage_analysis.json
  results/fig_coverage.png

Usage:
  python code/analyze_coverage.py [--sweep configs/sweep.yaml]
"""

import argparse
import csv

import numpy as np

import hw2  # noqa: F401
from hw2 import plotting, utils
from hw2.envs import pick_and_place_from_uniforms

RESULTS_DIR = utils.REPO_ROOT / "results"
DISTANCE_BINS = [0.0, 0.05, 0.08, 0.12, 0.40]   # m, nearest-demo distance bins for the figure
MIN_LAYOUTS_TO_PLOT = 5   # a bin needs at least this many test layouts to be drawn
EDGE_ANGLE_DEG = 45.0  # pick spots more than this far left/right of straight ahead count as "edge of the workspace"


def compute_layouts(seeds):
  """Pick and place spots (N, 2) for each seed, exactly as PandaPickAndPlace.reset() makes them."""
  import jax
  import jax.numpy as jnp

  def layout_for_key(random_key):
    _, layout_key = jax.random.split(random_key)   # same split as in reset()
    return pick_and_place_from_uniforms(jax.random.uniform(layout_key, (5,)), xp=jnp)

  random_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, dtype=jnp.int32))
  pick_xy, place_xy = jax.jit(jax.vmap(layout_for_key))(random_keys)
  return np.asarray(pick_xy), np.asarray(place_xy)


def load_per_seed_outcomes(run_names, eval_seeds):
  """Success rate and never-reached rate of every evaluation seed, averaged over the given runs."""
  seed_index = {seed: i for i, seed in enumerate(eval_seeds)}
  success = np.zeros(len(eval_seeds))
  never_reached = np.zeros(len(eval_seeds))
  for run_name in run_names:
    with open(RESULTS_DIR / "runs" / run_name / "episodes.csv") as f:
      for row in csv.DictReader(f):
        i = seed_index[int(row["seed"])]
        success[i] += (row["success"] == "True") / len(run_names)
        never_reached[i] += (row["failure_mode"] == "never_reached") / len(run_names)
  return success, never_reached


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--sweep", default="configs/sweep.yaml")
  parser.add_argument("--train-config", default="configs/dp_base.yaml")
  args = parser.parse_args()
  plotting.use_report_style()
  sweep = utils.load_config(args.sweep)
  eval_config = utils.load_config(args.train_config)["eval"]

  # Every baseline demo succeeded, so the n-demo subset is simply demo seeds 0 .. n-1.
  collection = utils.load_json(RESULTS_DIR / "demos" / "collection_baseline.json")
  assert collection["num_success"] == collection["num_attempts"], "some demo seeds failed; pick the subset by success"
  max_demos = max(condition["num_demos"] for condition in sweep["conditions"])
  demo_pick, demo_place = compute_layouts(np.arange(max_demos))
  eval_seeds = eval_config["seed_offset"] + np.arange(eval_config["num_episodes"])
  eval_pick, eval_place = compute_layouts(eval_seeds)
  demo_layouts = np.hstack([demo_pick, demo_place])
  eval_layouts = np.hstack([eval_pick, eval_place])
  is_edge_pick = np.abs(np.degrees(np.arctan2(eval_pick[:, 1], eval_pick[:, 0]))) > EDGE_ANGLE_DEG

  summary = {"distance_bins_m": DISTANCE_BINS, "edge_angle_deg": EDGE_ANGLE_DEG,
             "num_edge_eval_layouts": int(is_edge_pick.sum()), "conditions": {}}
  success_by_condition = {}
  fig, (left_ax, right_ax) = plotting.plt.subplots(1, 2, figsize=(11, 4.2))
  for color_index, condition in enumerate(sweep["conditions"]):
    num_demos = condition["num_demos"]
    run_names = [f"{condition['name']}_s{seed}" for seed in sweep["seeds"]]
    success, never_reached = load_per_seed_outcomes(run_names, eval_seeds)
    success_by_condition[num_demos] = success
    distances = np.linalg.norm(eval_layouts[:, None, :] - demo_layouts[None, :num_demos, :], axis=2)
    nearest_demo_distance = distances.min(axis=1)

    bins = []
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
      in_bin = (nearest_demo_distance >= low) & (nearest_demo_distance < high)
      bins.append({"low": low, "high": high, "num_layouts": int(in_bin.sum()),
                   "mean_distance": float(nearest_demo_distance[in_bin].mean()) if in_bin.any() else None,
                   "success_rate": float(success[in_bin].mean()) if in_bin.any() else None})
    summary["conditions"][condition["name"]] = {
        "num_demos": num_demos,
        "nearest_demo_distance_median_m": float(np.median(nearest_demo_distance)),
        "correlation_distance_vs_success": float(np.corrcoef(nearest_demo_distance, success)[0, 1]),
        "success_by_distance_bin": bins,
        "success_edge_pick": float(success[is_edge_pick].mean()),
        "success_center_pick": float(success[~is_edge_pick].mean()),
        "never_reached_edge_pick": float(never_reached[is_edge_pick].mean()),
        "never_reached_center_pick": float(never_reached[~is_edge_pick].mean()),
    }

    # Each point sits at the average distance of the layouts in its bin; bins with very few layouts are left out.
    plotted_bins = [b for b in bins if b["num_layouts"] >= MIN_LAYOUTS_TO_PLOT]
    left_ax.plot([100 * b["mean_distance"] for b in plotted_bins], [100 * b["success_rate"] for b in plotted_bins],
                 marker="o", color=plotting.COLORS[color_index],
                 label=f"{num_demos} demos (median distance {100 * np.median(nearest_demo_distance):.1f} cm)")

  left_ax.set_xlabel("distance from the test layout to the closest training demo [cm]")
  left_ax.set_ylabel("student success rate [%]")
  left_ax.set_ylim(0, 100)
  left_ax.set_title("Success drops on layouts far from the training demos")
  left_ax.legend(fontsize=8)

  # Right: where the 100-demo student fails, seen from the top (pick spots only).
  largest = max(success_by_condition)
  points = right_ax.scatter(eval_pick[:, 0], eval_pick[:, 1], c=100 * success_by_condition[largest], cmap="RdYlBu",
                            vmin=0, vmax=100, s=40, edgecolors=plotting.SECONDARY_TEXT_COLOR, linewidths=0.4,
                            label="test pick spot")
  right_ax.scatter(demo_pick[:, 0], demo_pick[:, 1], marker="x", s=14, color=plotting.SECONDARY_TEXT_COLOR,
                   label="training demo pick spot")
  right_ax.scatter([0], [0], marker="s", s=60, color=plotting.TEXT_COLOR, label="robot base")
  for sign in [-1, 1]:
    angle = np.radians(sign * EDGE_ANGLE_DEG)
    right_ax.plot([0, 0.65 * np.cos(angle)], [0, 0.65 * np.sin(angle)], ls="--", lw=1, color=plotting.REFERENCE_COLOR)
  right_ax.set_aspect("equal")
  right_ax.set_xlabel("x [m]")
  right_ax.set_ylabel("y [m]")
  right_ax.set_title(f"{largest}-demo student: success per test layout (dashed = ±{EDGE_ANGLE_DEG:.0f}°)")
  right_ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
  fig.colorbar(points, ax=right_ax, label="success over 3 seeds [%]")
  plotting.save_figure(fig, RESULTS_DIR / "fig_coverage.png")

  utils.save_json(summary, RESULTS_DIR / "coverage_analysis.json")
  for name, result in summary["conditions"].items():
    rates = ", ".join(f"{100 * b['low']:.0f}-{100 * b['high']:.0f} cm: "
                      + ("-" if b["success_rate"] is None else f"{100 * b['success_rate']:.0f}% (n={b['num_layouts']})")
                      for b in result["success_by_distance_bin"])
    print(f"{name}: {rates} | edge picks: success {100 * result['success_edge_pick']:.0f}%, "
          f"never reached {100 * result['never_reached_edge_pick']:.0f}% "
          f"(center: {100 * result['success_center_pick']:.0f}%, {100 * result['never_reached_center_pick']:.0f}%)")


if __name__ == "__main__":
  main()
