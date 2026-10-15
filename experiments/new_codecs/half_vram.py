"""Conservative PyTorch allocator cap plus whole-device memory monitoring.

Use one experiment process per physical GPU. This is not a CUDA partition: an
unrelated process can change usage between samples; stop our process if it does.
"""
import json
import os
from pathlib import Path
import subprocess
import threading


def allocator_budget(total_mib, used_mib):
    # Leave room for CUDA context/libraries and for unrelated services.
    budget = min(18 * 1024, total_mib * 0.5 - used_mib - 2048)
    if budget < 1024:
        raise RuntimeError("Insufficient headroom under half-device VRAM limit")
    return int(budget)


def install(log_path):
    import torch
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical.isdigit():
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index")

    def read():
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", physical, "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        return [int(v.strip()) for v in raw.strip().split(",")]

    total, used = read()
    budget = allocator_budget(total, used)
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.set_num_threads(2)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    state = {"total_mib": total, "initial_used_mib": used, "allocator_cap_mib": budget,
             "device_limit_mib": total / 2, "physical_gpu": physical}
    path.with_suffix(".json").write_text(json.dumps(state, indent=2))
    print("VRAM guard", state, flush=True)

    def watch():
        with path.open("a", buffering=1) as f:
            while not stop.is_set():
                try:
                    _, current = read()
                    f.write(json.dumps({"used_mib": current}) + "\n")
                    if current >= total / 2 - 512:
                        print("VRAM guard: stopping own process before half-device threshold", flush=True)
                        os._exit(75)
                except Exception as e:
                    print("VRAM monitoring failed; stopping own process:", e, flush=True)
                    os._exit(76)
                stop.wait(0.25)

    threading.Thread(target=watch, daemon=True).start()
    return stop


if __name__ == "__main__":
    assert allocator_budget(46068, 1517) == 18432
    assert allocator_budget(46068, 9000) == 11986
    try:
        allocator_budget(46068, 22000)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Should refuse allocation with insufficient headroom")
    print("VRAM budget checks passed")
