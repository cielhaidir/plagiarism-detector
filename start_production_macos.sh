#!/bin/bash

# macOS-specific production deployment script for plagiarism detection API
# Addresses PyTorch/sentence-transformers fork() issues on macOS

echo "Starting plagiarism detection API on macOS..."

# Set environment variables to prevent fork() issues
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install/upgrade dependencies with specific versions for compatibility
echo "Installing dependencies..."
pip install --upgrade -r requirements.txt

# Check if indices exist
if [ ! -d "indices" ] || [ ! -f "skripsi_with_skema.csv" ]; then
    echo "Warning: Required data files (indices/ or skripsi_with_skema.csv) not found!"
    echo "Please ensure all required data files are present before starting the server."
    exit 1
fi

echo "Starting Gunicorn server with macOS-specific configuration..."
echo "Note: Using single worker with fork-safety disabled for macOS compatibility"
export PYTHONPATH=$PWD:$PYTHONPATH
gunicorn --config gunicorn_macos.conf.py app:app