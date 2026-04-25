#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time

def print_health_check():
    import psutil
    import torch
    
    print("========================================")
    print(" DeltaStream Health Check")
    print("========================================")
    
    # System
    env = "Bare Metal Linux"
    try:
        with open("/proc/version", "r") as f:
            v = f.read().lower()
            if "microsoft" in v:
                env = "WSL2"
    except Exception:
        env = "Unknown Linux"
    print(f"System:    {env}")
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        print(f"GPU:       {gpu_name}, {vram_gb:.1f}GB VRAM, CUDA {cuda_ver}")
    else:
        print("GPU:       None / CUDA unavailable")
        
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM:       {ram.total/(1024**3):.1f}GB total, {ram.available/(1024**3):.1f}GB available")
    
    # Storage
    free_gb = shutil.disk_usage(".").free / (1024**3)
    print(f"Storage:   {free_gb:.1f} GB free")
    
    # io_uring
    io_uring = "blocked" if env == "WSL2" else "available"
    try:
        import liburing
    except ImportError:
        io_uring = "unavailable (liburing not installed)"
    print(f"io_uring:  {io_uring}")
    
    # mlock
    mlock = os.popen("ulimit -l").read().strip()
    if mlock == "unlimited":
        print("mlock:     unlimited")
    else:
        print(f"mlock:     limited ({mlock})")
        print("           -> FIX: echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.conf")
        
    print("\nRecommended model for this hardware:")
    if ram.total / (1024**3) >= 16:
        print("  meta-llama/Meta-Llama-3-8B-Instruct")
    elif ram.total / (1024**3) >= 8:
        print("  google/gemma-2-2b-it")
    else:
        print("  TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("========================================")

def check_disk_space(model_id):
    try:
        from huggingface_hub import model_info
        print(f"[*] Fetching model info for {model_id}...")
        info = model_info(model_id)
        # Sum safetensors files
        size_bytes = sum(f.size for f in info.siblings if f.rfilename.endswith(".safetensors") or f.rfilename.endswith(".bin"))
        if size_bytes == 0:
            size_bytes = sum(getattr(f, 'size', 0) or 0 for f in info.siblings)
            
        if size_bytes > 0:
            free_bytes = shutil.disk_usage(".").free
            # Need space for raw weights (cache) + delta format (~1x for delta due to compression, so ~2.2x total)
            needed = size_bytes * 2.2
            if free_bytes < needed:
                print(f"⚠️ WARNING: Model '{model_id}' requires ~{needed/(1024**3):.1f}GB disk space (base + delta).")
                print(f"   You only have {free_bytes/(1024**3):.1f}GB free.")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != 'y':
                    sys.exit(1)
            else:
                print(f"[*] Disk space OK (requires ~{needed/(1024**3):.1f}GB, {free_bytes/(1024**3):.1f}GB free).")
    except Exception as e:
        print(f"[*] Could not automatically verify disk space: {e}")

def run_chat(model_id, delta_dir, ram_gb, use_cpu):
    # 1. Check space
    check_disk_space(model_id)
    
    # 2 & 3. Convert if missing
    if not os.path.exists(delta_dir) or not os.path.exists(os.path.join(delta_dir, "manifest.json")):
        print(f"\n[*] Delta format not found at '{delta_dir}'.")
        print("[*] Starting DeltaStream conversion (this will download the model if not cached)...")
        cmd = [sys.executable, "-m", "deltastream", "convert", "--model", model_id, "--output", delta_dir, "--compress"]
        subprocess.run(cmd, check=True)
        print("[*] Conversion complete.\n")
        
    # 4. Initialize
    print("[*] Initializing DeltaStream Runtime...")
    from deltastream.runtime import DeltaStreamRuntime
    import torch
    import psutil
    
    runtime = DeltaStreamRuntime(
        model_id=model_id,
        delta_dir=delta_dir,
        max_ram_gb=ram_gb,
        compress=True,
    )
    
    if use_cpu:
        runtime.device = "cpu"
        
    tokenizer = runtime.tokenizer
    
    # 5. Chat loop
    print("\n" + "="*50)
    print(" DeltaStream Interactive Chat")
    print(f" Model: {model_id}")
    print(" Commands: /stats (performance), /clear (reset context), /exit")
    print("="*50 + "\n")
    
    messages = []
    last_gen_tokens = 0
    last_time = 0
    
    while True:
        try:
            user_input = input("User > ")
            if user_input.strip() == "":
                continue
            if user_input.strip() == "/exit":
                print("Exiting...")
                break
            if user_input.strip() == "/clear":
                messages = []
                print("Context cleared.")
                continue
            if user_input.strip() == "/stats":
                ram_used = psutil.virtual_memory().used / (1024**3)
                vram_used = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                tok_sec = (last_gen_tokens / last_time) if last_time > 0 else 0
                print(f"[Stats] Speed: {tok_sec:.2f} tok/s | RAM: {ram_used:.1f} GB | VRAM: {vram_used:.1f} GB")
                continue
                
            messages.append({"role": "user", "content": user_input})
            
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    input_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nAssistant: "
            else:
                input_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nAssistant: "
                
            input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
            input_len = input_ids.shape[1]
            
            print("Assistant > ", end="", flush=True)
            
            start_t = time.time()
            out_ids = runtime.generate(input_ids, max_new_tokens=512)
            end_t = time.time()
            
            gen_ids = out_ids[0][input_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            print(response + "\n")
            
            messages.append({"role": "assistant", "content": response.strip()})
            
            last_gen_tokens = len(gen_ids)
            last_time = end_t - start_t
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")

def main():
    parser = argparse.ArgumentParser(description="DeltaStream Unified Runner")
    parser.add_argument("--model", type=str, help="HuggingFace model ID or local path")
    parser.add_argument("--delta", type=str, help="Custom delta output path (optional)")
    parser.add_argument("--ram", type=float, default=10.0, help="RAM budget in GB (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only execution")
    parser.add_argument("--check", action="store_true", help="Run system health check")
    
    args = parser.parse_args()
    
    if args.check:
        print_health_check()
        sys.exit(0)
        
    if not args.model:
        print("❌ Error: --model is required unless running --check")
        print("Usage: python run.py --model google/gemma-2-2b-it")
        sys.exit(1)
        
    delta_dir = args.delta if args.delta else f"delta_{args.model.split('/')[-1]}"
    run_chat(args.model, delta_dir, args.ram, args.cpu)

if __name__ == "__main__":
    main()
