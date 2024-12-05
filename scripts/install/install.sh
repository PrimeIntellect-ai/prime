#!/usr/bin/env bash
 
# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

main() {
    # Check if sudo is installed
    if ! command -v sudo &> /dev/null; then
        apt update || true
        apt install sudo -y || true
    fi

    log_info "Updating apt..."
    sudo apt update || true

    log_info "Installing cmake python3-dev..."
    sudo apt install python3-dev cmake -y || true

    log_info "Installing iperf..."
    sudo apt install iperf -y || true

    log_info "Cloning repository..."
    git clone https://github.com/PrimeIntellect-ai/prime.git || true
    
    log_info "Entering project directory..."
    cd prime || true
    
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
    
    log_info "Sourcing uv environment..."
    source $HOME/.local/bin/env || true
        
    log_info "Creating virtual environment..."
    uv venv || true
    
    log_info "Activating virtual environment..."
    source .venv/bin/activate || true
    
    log_info "Installing dependencies..."
    uv sync --extra all || true
    
    log_info "Installing flash-attn..."
    uv pip install flash-attn==2.6.3 --no-build-isolation || true
    
    log_info "Updating git submodules..."
    git submodule update --init --recursive || true
    
    log_info "Downloading data..."
    mkdir -p datasets || true
    uv run python scripts/subset_data.py --dataset_name PrimeIntellect/fineweb-edu --data_world_size 1 --data_rank 0 --max_shards 128 || true
    mv fineweb-edu/ datasets/fineweb-edu/ || true

    log_info "Installation completed! You can double check that everything is install correctly by running 'GLOO_SOCKET_IFNAME=lo GLOBAL_ADDR=localhost GLOBAL_RANK=0 GLOBAL_UNIQUE_ID=0 GLOBAL_WORLD_SIZE=1 GLOBAL_PORT=8989  uv run torchrun --nproc_per_node=2 src/zeroband/train.py  @configs/debug/diloco.toml'"
}

main