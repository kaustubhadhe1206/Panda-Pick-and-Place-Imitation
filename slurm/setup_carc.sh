#!/bin/bash
# One-time environment setup on USC CARC Discovery.
# Run on a LOGIN node (needs internet):   bash slurm/setup_carc.sh
# Creates a Python 3.12 venv with uv, installs the pinned requirements + PyTorch,
# registers a Jupyter kernel "Python (hw2)" for CARC OnDemand, and pre-downloads
# the MuJoCo Menagerie assets that MuJoCo Playground fetches on first use
# (GPU compute nodes may not have outbound internet).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${HW2_VENV:-$HOME/.venvs/hw2}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
# One resolve for everything so torch's and jax's CUDA 12 libraries agree. The default PyPI
# torch wheel is built for CUDA 13, so take the CUDA 12.6 build (same torch version as on Colab).
uv pip install -r requirements.txt torch==2.11.0 ipykernel \
  --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
python -m ipykernel install --user --name hw2 --display-name "Python (hw2)"

JAX_PLATFORMS=cpu python -c "
import sys; sys.path.insert(0, 'code')
import hw2
from hw2.envs import load_pick_place_env
env = load_pick_place_env(); print('pick-and-place env loaded, action size', env.action_size)
"
echo "Done. Activate with: source $VENV/bin/activate"
