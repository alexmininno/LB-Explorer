#!/bin/bash

echo "========================================================"
echo "  Setting up Conda Environment for CY Explorer"
echo "========================================================"

# 1. Check for Conda
if ! command -v conda &> /dev/null
then
    echo "[ERROR] Conda could not be found."
    echo "Please install Anaconda or Miniconda first."
    echo "If installed, you might need to run: source ~/anaconda3/etc/profile.d/conda.sh"
    exit 1
fi

# 2. Initialize Conda for this script shell
# This is a common fix for 'conda activate' errors in scripts
eval "$(conda shell.bash hook)"

# 3. Create Environment
echo ""
echo "[*] Creating environment 'cy-explorer'..."
conda create -n cy-explorer python=3.9 -y

# 4. Activate
echo ""
echo "[*] Activating 'cy-explorer'..."
conda activate cy-explorer

# 5. Install PyTorch
echo ""
echo "[*] Installing PyTorch..."
# Standard install. If you are on a Mac with M1/M2 chip, this usually defaults to CPU or MPS automatically.
# If on Linux with NVIDIA, this grabs the CUDA version.
conda install pytorch torchvision torchaudio -c pytorch -c nvidia -y

# 6. Install dependencies
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found! Installing basic packages..."
    pip install numpy tqdm sympy matplotlib pycicy
fi


# 7. Verify
echo ""
echo "[*] Verifying..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo ""
echo "========================================================"
echo "  Setup Complete!"
echo "  To start, run in your terminal:"
echo "      conda activate cy-explorer"
echo "========================================================"