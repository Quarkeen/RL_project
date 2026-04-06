#!/usr/bin/env bash
# =============================================================================
#  F1TENTH PPO Racing Agent — Environment Bootstrap
#  Initializes a conda env, clones f1tenth_gym, and installs the RL stack.
# =============================================================================
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ENV_NAME="f1tenth_rl"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_DIR="${PROJECT_DIR}/f1tenth_gym"

# ── Color helpers ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Step 1: Verify conda is available ─────────────────────────────────────────
info "Checking for conda installation..."
if ! command -v conda &>/dev/null; then
    fail "conda not found. Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
fi
ok "conda found at $(which conda)"

# ── Step 2: Create / reuse conda environment ──────────────────────────────────
if conda env list | grep -qw "${ENV_NAME}"; then
    warn "Conda environment '${ENV_NAME}' already exists — reusing."
else
    info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    ok "Environment '${ENV_NAME}' created."
fi

# ── Step 3: Activate the environment ──────────────────────────────────────────
info "Activating conda environment '${ENV_NAME}'..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
ok "Active Python: $(python --version) @ $(which python)"

# ── Step 4: Clone f1tenth_gym ─────────────────────────────────────────────────
if [ -d "${GYM_DIR}" ]; then
    warn "f1tenth_gym directory already exists at ${GYM_DIR} — skipping clone."
else
    info "Cloning f1tenth_gym repository..."
    git clone https://github.com/f1tenth/f1tenth_gym.git "${GYM_DIR}"
    ok "f1tenth_gym cloned."
fi

# ── Step 5: Install f1tenth_gym in editable mode ──────────────────────────────
#   gym==0.19.0 has incompatibilities with modern Python tooling:
#     - Malformed specifier "opencv-python>=3." rejected by pip>=24.1
#     - extras_require format broken with setuptools>=67
#   Solution: temporarily downgrade pip + setuptools, install gym, then restore.
info "Saving current pip version..."
ORIG_PIP_VER=$(pip --version | awk '{print $2}')

info "Downgrading pip<24.1 + setuptools<66 for gym==0.19.0 compatibility..."
pip install "pip<24.1" "setuptools<66"

info "Pre-installing gym==0.19.0..."
pip install "gym==0.19.0"

info "Installing f1tenth_gym in editable mode..."
pip install -e "${GYM_DIR}"

info "Restoring pip, setuptools, packaging, and wheel to latest versions..."
pip install --upgrade pip setuptools packaging wheel
ok "f1tenth_gym installed."

# ── Step 6: Install the RL stack ──────────────────────────────────────────────
info "Installing RL stack (PyTorch, Gymnasium, TensorBoard, etc.)..."
pip install \
    torch \
    gymnasium \
    tensorboard \
    pyyaml \
    "numpy<2.0" \
    opencv-python \
    matplotlib \
    pyglet==1.5.20
ok "RL stack installed."

# ── Step 7: Verify critical imports ──────────────────────────────────────────
info "Running import sanity checks..."
python -c "
import sys, importlib
packages = ['torch', 'gymnasium', 'tensorboard', 'yaml', 'numpy', 'cv2', 'f110_gym']
failed = []
for p in packages:
    try:
        mod = importlib.import_module(p)
        ver = getattr(mod, '__version__', 'n/a')
        print(f'  ✓ {p:20s}  {ver}')
    except ImportError as e:
        print(f'  ✗ {p:20s}  FAILED ({e})')
        failed.append(p)
if failed:
    sys.exit(1)
"
ok "All imports verified."

# ── Step 8: Create project directory structure ────────────────────────────────
info "Ensuring project directory layout..."
mkdir -p "${PROJECT_DIR}/runs"        # TensorBoard log directory
mkdir -p "${PROJECT_DIR}/checkpoints" # Model checkpoints
mkdir -p "${PROJECT_DIR}/trajectories" # CSV trajectory output
ok "Directory structure ready."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     F1TENTH RL Environment Setup Complete!                ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Activate with:  conda activate ${ENV_NAME}               ║${NC}"
echo -e "${GREEN}║  Train with:     python train.py                         ║${NC}"
echo -e "${GREEN}║  TensorBoard:    tensorboard --logdir=runs/              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
