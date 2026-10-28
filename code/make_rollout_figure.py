"""Frames from student rollout videos, one row per episode (Part 4 failure analysis, report Figure 6).

Takes one frame per second from a few of the evaluation videos written by
run_sweep.py (videos/student/<run>/) and puts them in a grid: a baseline
success, two baseline failures and a 25-demo failure. Runs on CPU in a few seconds.

Writes:
  results/fig_student_rollouts.png

Usage:
  python code/make_rollout_figure.py
"""

import imageio.v2 as imageio

import hw2  # noqa: F401
from hw2 import plotting, utils

VIDEO_DIR = utils.REPO_ROOT / "videos" / "student"
FRAME_TIMES_S = [0, 1, 2, 3, 4, 5]
CROP_ROWS, CROP_COLUMNS = slice(40, None), slice(60, 580)   # cut away empty floor around the robot
# (row label, video): the episodes shown in the report
ROWS = [
    ("(a) 100 demos\nseed 100000\nsuccess", "n100_s0/success_seed100000.mp4"),
    ("(b) 100 demos\nseed 100001\nmisplaced\n(7.2 cm off)", "n100_s0/failure_seed100001.mp4"),
    ("(c) 100 demos\nseed 100005\ndrops cube,\nthen diverges", "n100_s0/failure_seed100005.mp4"),
    ("(d) 25 demos\nseed 100001\nnever reached\nthe cube", "n25_s0/failure_seed100001.mp4"),
]


def main():
  plotting.use_report_style()
  fig, axes = plotting.plt.subplots(len(ROWS), len(FRAME_TIMES_S), figsize=(15, 8.4),
                                    gridspec_kw=dict(wspace=0.04, hspace=0.06))
  for row, (label, video_name) in enumerate(ROWS):
    reader = imageio.get_reader(VIDEO_DIR / video_name)
    fps = reader.get_meta_data()["fps"]
    frames = [frame for frame in reader]
    reader.close()
    for column, time_s in enumerate(FRAME_TIMES_S):
      ax = axes[row, column]
      frame_index = min(int(round(time_s * fps)), len(frames) - 1)
      ax.imshow(frames[frame_index][CROP_ROWS, CROP_COLUMNS])
      ax.set_xticks([])
      ax.set_yticks([])
      ax.grid(False)
      if row == 0:
        ax.set_title(f"t = {time_s} s")
    axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8)
  path = plotting.save_figure(fig, utils.REPO_ROOT / "results" / "fig_student_rollouts.png")
  print("wrote", path.relative_to(utils.REPO_ROOT).as_posix())


if __name__ == "__main__":
  main()
