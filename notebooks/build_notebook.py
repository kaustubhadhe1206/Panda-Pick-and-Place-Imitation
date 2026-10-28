"""Generates notebooks/run_pipeline.ipynb (keeps the notebook diff-able as Python).

  python notebooks/build_notebook.py

Saved outputs are kept: a code cell whose source is unchanged keeps the outputs it
has in the existing notebook, so regenerating never throws away results.
"""

import json
from pathlib import Path

cells = []


def md(text):
  cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
  cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# CSCI 645 HW2: Imitation learning for robot pick-and-place (Diffusion Policy)

The saved outputs in this notebook are from the run used in the report (Google Colab, one T4 GPU).

Run it **either** on Google Colab (T4) **or** on a CARC GPU node (OnDemand JupyterLab with the
`Python (hw2)` kernel from `slurm/setup_carc.sh`). Don't mix platforms inside one sweep, because
GPU simulation differs slightly across GPU types.

Every cell only calls the scripts in `code/`, so the notebook does exactly what the command-line /
SLURM workflow does. Steps are **resumable**: re-running a cell skips work that is already done.

| Section | What it does | Homework part |
|---|---|---|
| 0. Setup | settings, clone, Google Drive, install | – |
| 1. Checks | unit tests, smoke test, training-speed benchmark | – |
| 2. Demonstrations | collect 100 expert demos; statistics, figures, videos | Part 1 |
| 3. Teacher evaluation | expert on the 100 held-out test seeds | Parts 1, 4 |
| 4. Sanity check (optional) | overfit 5 demos | – |
| 5. Training sweep | 100 / 50 / 25 demos × 3 seeds: train + evaluate | Parts 2–3 |
| 6. Results | baseline vs. ablation table and comparison figures | Part 4 |
| 7. Failure analysis | layout coverage and rollout frames (CPU only) | Part 4 |
| 8. Save code edits | commit edits made on the runtime | – |

**Storage (Colab):** all outputs go to `My Drive/CSCI-645 Homework 2/pick-and-place-swing/`. That covers checkpoints and raw rollouts in `runs/`,
copies of `data/`, `results/` and `videos/`, and a log of every command in `logs/`. A disconnect therefore never
loses a trained model. Python packages and the git clone live on the VM and are reinstalled each session.

**Rules for edits:** change behaviour through `--set key=value` overrides where possible. If you
edit a `.py` file on the runtime, commit + push it (section 8) before the session ends.
""")

md(r"""
## 0. Setup (run every session)
Settings, clone / pull the repo, mount Google Drive and install packages. The clone cell asks for an
optional GitHub token, only needed to push results back to the repo (press Enter to skip).
""")

code(r"""
# ---- 0. Settings ----------------------------------------------------------------------------
REPO_URL = "https://github.com/kaustubhadhe1206/Panda-Pick-and-Place-Imitation.git"   # never put a token here
BRANCH = "main"
GIT_NAME, GIT_EMAIL = "kashinadhe", "kaustubhadhe@gmail.com"
DRIVE_FOLDER = "CSCI-645 Homework 2/pick-and-place-swing"   # Colab: all outputs are stored in My Drive/<this folder>
# (the older "pick-and-place" folder holds results from the earlier layout; it is not used anymore)

import os, sys, subprocess, pathlib, shlex
IN_COLAB = os.path.isdir("/content") and not os.environ.get("SLURM_JOB_ID")
cwd = pathlib.Path.cwd()
if (cwd / "code" / "hw2").is_dir():
    REPO = cwd
elif (cwd.parent / "code" / "hw2").is_dir():
    REPO = cwd.parent                                  # notebook opened from notebooks/ (CARC)
else:
    REPO = pathlib.Path("/content" if IN_COLAB else pathlib.Path.home()) / pathlib.Path(REPO_URL).stem
print("platform:", "Colab" if IN_COLAB else "CARC/other", "| repo:", REPO)
""")

code(r"""
# ---- 1. Clone / pull (public repo: no token needed) -----------------------------------------
# A token is OPTIONAL and only used to push results (CSVs, figures, videos) back to GitHub, so
# they show up in the repo without downloading them from Drive. Press Enter to skip it.
# If used: fine-grained token for this one repo, permission "Contents: Read and write".
from getpass import getpass
if "GH_TOKEN" not in os.environ:
    os.environ["GH_TOKEN"] = getpass("GitHub token for pushing results (Enter to skip): ").strip()
TOKEN = os.environ["GH_TOKEN"]
CAN_PUSH = bool(TOKEN)
if CAN_PUSH:
    os.environ["HW2_PUSH_URL"] = REPO_URL.replace("https://", f"https://x-access-token:{TOKEN}@")
else:
    os.environ.pop("HW2_PUSH_URL", None)               # run_sweep.py --push then skips pushing
print("GitHub push:", "enabled" if CAN_PUSH else "disabled (outputs are kept on Drive only)")

def run_git(*args, check=True):
    # Run a git command and print its output (with the token hidden).
    result = subprocess.run(["git", *args], cwd=REPO if REPO.exists() else None, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if TOKEN: output = output.replace(TOKEN, "***")
    if output.strip(): print(output.strip())
    if check and result.returncode: raise RuntimeError(f"git {args[0]} failed")
    return result

if "<" in REPO_URL:
    raise ValueError("Set REPO_URL in cell 0 to your GitHub repo first.")
if not (REPO / ".git").exists():
    run_git("clone", "-q", "-b", BRANCH, REPO_URL, str(REPO))
else:
    run_git("pull", "--ff-only", REPO_URL, BRANCH)
if CAN_PUSH:
    run_git("config", "user.name", GIT_NAME); run_git("config", "user.email", GIT_EMAIL)
os.chdir(REPO)
run_git("log", "--oneline", "-3")
""")

code(r"""
# ---- 1b. Google Drive storage (Colab only) --------------------------------------------------
# My Drive/CSCI-645 Homework 2/pick-and-place-swing/
#   runs/      checkpoints + raw rollouts (repo's runs/ is a symlink here, so they survive disconnects)
#   data/ results/ videos/   copies of everything the pipeline produces (refreshed after each step)
#   logs/      full console output of every pipeline command
# Packages and the git clone stay on the VM disk (Drive is too slow for imports / git).
import shutil, datetime
STORAGE = None
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        STORAGE = pathlib.Path("/content/drive/MyDrive") / DRIVE_FOLDER
    except Exception as error:
        print("WARNING: Google Drive could not be mounted (" + repr(error)[:200] + ").\n"
              "Outputs stay on the VM" + ("; results are still pushed to GitHub after each step." if CAN_PUSH
              else " and are LOST on disconnect. Re-run cell 1 with a GitHub token to push them."))
if STORAGE is not None:
    for folder in ["runs", "logs", "data", "results", "videos"]:
        (STORAGE / folder).mkdir(parents=True, exist_ok=True)
    runs_dir = REPO / "runs"
    if runs_dir.exists() and not runs_dir.is_symlink():   # move anything made before Drive was mounted
        shutil.copytree(runs_dir, STORAGE / "runs", dirs_exist_ok=True)
        shutil.rmtree(runs_dir)
    if not runs_dir.is_symlink():
        runs_dir.symlink_to(STORAGE / "runs", target_is_directory=True)
    os.environ["HW2_MIRROR_DIR"] = str(STORAGE)            # run_sweep.py copies each finished run here
    print("Drive storage:", STORAGE)
    saved_checkpoints = sorted(path.parent.name for path in (STORAGE / "runs").glob("*/checkpoint.pt"))
    print("existing checkpoints:", saved_checkpoints)

def copy_outputs_to_drive():
    # Copy data/, results/, videos/ from the repo into the Drive folder (does nothing without Drive).
    if STORAGE is None: return
    for folder in ["data", "results", "videos"]:
        if (REPO / folder).exists():
            shutil.copytree(REPO / folder, STORAGE / folder, dirs_exist_ok=True)
    print("saved to Drive:", STORAGE)
""")

code(r"""
# ---- 2. Install (Colab only; CARC uses the venv built by slurm/setup_carc.sh) ----------------
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    # Note: later, JAX may print "PJRT_Api already exists for device type cuda". That is harmless:
    # Colab already ships a JAX CUDA plugin and pip adds a second copy; JAX just uses the first one.
    # EGL vendor file for headless MuJoCo rendering on Colab GPUs (as in the Playground notebooks).
    egl_vendor_file = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(egl_vendor_file):
        os.makedirs(os.path.dirname(egl_vendor_file), exist_ok=True)
        with open(egl_vendor_file, "w") as f:
            f.write('{"file_format_version" : "1.0.0", "ICD" : {"library_path" : "libEGL_nvidia.so.0"}}')
os.environ["MUJOCO_GL"] = "egl"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])

def run_command(command):
    # Run a pipeline command from the repo root, show its output live (and save it to a log on Drive).
    print("$", command, flush=True)
    log_file = None
    if STORAGE is not None:
        script_name = "-".join(pathlib.Path(arg).stem for arg in shlex.split(command)[1:3])
        log_file = open(STORAGE / "logs" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{script_name}.log", "w")
        log_file.write(f"$ {command}\n")
    process = subprocess.Popen(shlex.split(command), cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
        if log_file: log_file.write(line); log_file.flush()
    exit_code = process.wait()
    if log_file: log_file.close()
    if exit_code: raise RuntimeError(f"command failed ({exit_code}): {command}")

def save_and_push(paths, message):
    # Copy outputs to Drive, then (if a token was given) commit the given paths and push to GitHub.
    copy_outputs_to_drive()
    if not CAN_PUSH: return print("GitHub push skipped (no token)")
    paths = [path for path in paths if (REPO / path).exists()]
    if not paths: return print("nothing to push")
    run_git("add", "-f", *paths)
    if run_git("diff", "--cached", "--quiet", check=False).returncode:
        run_git("commit", "-q", "-m", message)
        run_git("push", "-q", os.environ["HW2_PUSH_URL"], f"HEAD:{BRANCH}")
        print("pushed:", message)
    else:
        print("nothing to commit")

from IPython.display import Image, Video, display, Markdown
""")

md(r"""
## 1. Checks
### Unit tests and smoke test
CPU unit tests, then versions, GPU, environment sizes, expert success on 50 seeds, a rendered video
and a speed estimate (summarized in the final `SMOKE TEST REPORT` block). To tune the expert, re-run
with overrides, e.g.
`run_command("python code/smoke_test.py --checks env expert render --set expert.approach_speed=0.025")`.
""")

code(r"""
run_command("python tests/test_cpu.py")
run_command("python code/smoke_test.py")
""")

md(r"""
### Training speed benchmark (~2 min)
Compares speed settings on this GPU (random data, no simulator), summarized in the `TRAINING SPEED`
block. All runs in the report use the default fp32. A different setting must be used for the *whole*
sweep (`EXTRA_SETTINGS` in section 5), never for only some runs.
""")

code(r"""
run_command("python code/benchmark_training.py")
""")

md(r"""
### Smoke-test videos
Expert videos rendered by the smoke test above.
""")

code(r"""
for video_path in ["videos/smoke/expert_smoke.mp4", "videos/smoke/expert_smoke_failure.mp4"]:
    if (REPO / video_path).exists():
        display(Markdown(f"`{video_path}`"), Video(str(REPO / video_path), embed=True, width=480))
""")

md(r"""
## 2. Demonstrations (Part 1)
Collects 100 successful expert demos (env seeds 0–99), then makes the statistics, figures and videos
used in report Q1–Q2. Collection is skipped if `data/demos_baseline.npz` already exists (set
`FORCE = True` to collect again).
""")

code(r"""
FORCE = False
if FORCE or not (REPO / "data/demos_baseline.npz").exists():
    run_command("python code/collect_demos.py")
run_command("python code/inspect_demos.py")
for figure_name in ["fig_frames", "fig_coverage", "fig_trajectories", "fig_actions", "fig_phases"]:
    figure_path = REPO / f"results/demos/{figure_name}.png"
    if figure_path.exists(): display(Image(str(figure_path)))
grid_video = REPO / "videos/expert/expert_demos_grid.mp4"
if grid_video.exists(): display(Video(str(grid_video), embed=True, width=640))
save_and_push(["data/demos_baseline.npz", "results/demos", "videos/expert"], "demos: baseline dataset + inspection")
""")

md(r"""
## 3. Teacher evaluation
Runs the expert on the 100 held-out test seeds (100000–100099), the same seeds every student is
tested on.
""")

code(r"""
if not (REPO / "results/teacher_eval/eval_summary.json").exists():
    run_command("python code/evaluate.py --teacher --out-dir results/teacher_eval --video-dir videos/teacher_eval")
save_and_push(["results/teacher_eval", "videos/teacher_eval"], "results: teacher evaluation")
""")

md(r"""
## 4. Sanity check (optional)
Train on 5 demos and evaluate **on those same 5 seeds**. Near-perfect success here means the
data -> training -> execution path is wired correctly. Not part of the reported results.
""")

code(r"""
RUN_SANITY = True
if RUN_SANITY:
    run_command("python code/train.py --run-dir runs/sanity_n5 --set data.num_demos=5 train.num_steps=3000 eval.batch_size=5")
    run_command("python code/evaluate.py --run-dir runs/sanity_n5 --on-train-seeds --out-dir results/sanity_n5 --num-videos 1")
""")

md(r"""
## 5. Training sweep (Parts 2–3, resumable)
Runs `configs/sweep.yaml` (100 / 50 / 25 demos x seeds 0, 1, 2): train, then evaluate on the 100
test seeds. Each finished run is copied to Drive (and pushed to GitHub if a token was given).
Training saves a resumable state to Drive every 2000 steps (`runs/<run>/train_state.pt`), and a
finished run is never retrained. If the session dies, reconnect, run section 0, then this cell
again: finished runs are skipped and the interrupted run continues from its last saved step.
""")

code(r"""
# Settings applied to EVERY run of the sweep, e.g. "--set train.use_amp=true" or "--set train.num_steps=10000".
# Decide before the first run and do not change them halfway, so all runs stay comparable.
EXTRA_SETTINGS = ""
run_command(f"python code/run_sweep.py --dry-run {EXTRA_SETTINGS}")
run_command(f"python code/run_sweep.py --push {EXTRA_SETTINGS}")
""")

md(r"""
## 6. Results (Part 4)
The baseline vs. ablation table, then success vs. number of demos, success vs. distance threshold,
failure modes, training loss and action smoothness.
""")

code(r"""
run_command("python code/analyze.py")
display(Markdown((REPO / "results/summary_table.md").read_text()))
for figure_name in ["fig_success_vs_demos", "fig_success_vs_threshold", "fig_failure_modes",
                    "fig_loss_curves", "fig_smoothness"]:
    display(Image(str(REPO / f"results/{figure_name}.png")))
save_and_push(["results", "videos"], "results: aggregate tables and figures")
""")

md(r"""
## 7. Failure analysis (Part 4)
Success vs. distance from each test layout to the closest training demo (`code/analyze_coverage.py`,
report Figure 9) and frames from student rollout videos (`code/make_rollout_figure.py`, report
Figure 6). CPU only, a few seconds; needs the results and videos from section 5.

*In the saved notebook, this cell's outputs come from running it on a laptop CPU after the Colab
session, using the same result files.*
""")

code(r"""
run_command("python code/analyze_coverage.py")
run_command("python code/make_rollout_figure.py")
for figure_name in ["fig_coverage", "fig_student_rollouts"]:
    display(Image(str(REPO / f"results/{figure_name}.png")))
save_and_push(["results"], "results: failure analysis figures")
""")

md(r"""
## 8. Save code edits
If you edited `.py` files on the runtime, review the changes and commit them before the session ends.
""")

code(r"""
# ---- Commit any code edits made on this runtime (review the diff first!) ---------------------
run_git("status", "--short")
# save_and_push(["code", "configs"], "code: edits made on the runtime")
""")

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
for index, cell in enumerate(notebook["cells"]):
  cell["id"] = f"cell-{index:02d}"
notebook_path = Path(__file__).with_name("run_pipeline.ipynb")

# Keep the saved outputs (and notebook metadata) of the existing notebook for unchanged code cells.
if notebook_path.exists():
  with open(notebook_path, encoding="utf-8") as f:
    old_notebook = json.load(f)
  notebook["metadata"] = {**notebook["metadata"], **old_notebook["metadata"]}
  old_code_cells = {"".join(cell["source"]): cell for cell in old_notebook["cells"] if cell["cell_type"] == "code"}
  for cell in cells:
    old_cell = old_code_cells.get("".join(cell["source"])) if cell["cell_type"] == "code" else None
    if old_cell is not None:
      cell["outputs"] = old_cell.get("outputs", [])
      cell["execution_count"] = old_cell.get("execution_count")
      cell["metadata"] = old_cell.get("metadata", {})

with open(notebook_path, "w", newline="\n") as f:
  json.dump(notebook, f, indent=1)
print(f"wrote {notebook_path} ({len(cells)} cells)")
