#!/bin/bash
set -e

echo "========================================"
echo "      DeltaStream Setup Script"
echo "========================================"

# Detect OS
if grep -qi "microsoft" /proc/version 2>/dev/null; then
    OS_ENV="WSL2"
else
    OS_ENV="Bare Metal Linux"
fi

echo "[*] Detected Environment: $OS_ENV"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "[*] Activating virtual environment..."
source .venv/bin/activate

echo "[*] Installing dependencies (this may take a minute)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q

# Run health checks using Python
echo ""
python3 << 'EOF'
import sys
try:
    import torch
    import safetensors
    import deltastream
    import psutil
    from transformers import AutoModelForCausalLM
    imports_ok = True
except ImportError as e:
    imports_ok = False
    print(f"❌ Failed to import: {e}")

if imports_ok:
    print("✅ Packages: torch, safetensors, transformers, deltastream installed")

try:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB)")
    else:
        print("❌ GPU: No NVIDIA GPU detected or CUDA not available.")
except Exception:
    print("❌ GPU: Check failed.")

try:
    import liburing
    import os
    env = os.popen("cat /proc/version").read().lower()
    if "microsoft" in env:
        print("✅ io_uring: blocked (WSL2 Hyper-V constraint)")
    else:
        print("✅ io_uring: available (bare metal)")
except ImportError:
    print("❌ io_uring: liburing not installed (run: pip install liburing)")
EOF

# Check mlock
ULIMIT_MEMLOCK=$(ulimit -l)
if [ "$ULIMIT_MEMLOCK" = "unlimited" ]; then
    echo "✅ mlock: unlimited"
else
    echo "❌ mlock: limited ($ULIMIT_MEMLOCK)"
    echo "   -> WARNING: DeltaStream requires unlocked memory for maximum IO performance."
    echo "   -> FIX: Run the following as root to fix it permanently:"
    echo "      echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.conf"
    echo "      echo '* soft memlock unlimited' | sudo tee -a /etc/security/limits.conf"
    echo "   -> Then reboot or re-login."
fi

echo "========================================"
echo "Ready. Run: python run.py --model google/gemma-2-2b-it"
