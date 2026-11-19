"""Conservative PyTorch allocator cap plus whole-device memory monitoring.

Default: one experiment process per physical GPU. AEIC_SHARED_SMALL=1 opts into
an 8 GiB allocator / 10 GiB process cap on a shared device, with free headroom.
This is not a CUDA partition; unrelated processes can change between samples.
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


def shared_budget(total_mib, used_mib):
    # Opt-in AEIC pilot: 8 GiB allocator, 2 GiB context room, 1.5 GiB free.
    if total_mib - used_mib < 8 * 1024 + 2048 + 1536:
        raise RuntimeError("Insufficient free VRAM for the 8 GiB shared-GPU pilot")
    return 8 * 1024


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
    shared = os.environ.get("AEIC_SHARED_SMALL") == "1"
    budget = shared_budget(total, used) if shared else allocator_budget(total, used)
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.set_num_threads(2)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    state = {"total_mib": total, "initial_used_mib": used, "allocator_cap_mib": budget,
             "device_limit_mib": None if shared else total / 2, "physical_gpu": physical,
             "shared_small": shared, "process_limit_mib": 10240 if shared else None}
    path.with_suffix(".json").write_text(json.dumps(state, indent=2))
    print("VRAM guard", state, flush=True)

    def watch():
        with path.open("a", buffering=1) as f:
            while not stop.is_set():
                try:
                    _, current = read()
                    own = 0
                    if shared:
                        processes = subprocess.check_output(["nvidia-smi", "-i", physical,
                            "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                            text=True, timeout=5)
                        for line in processes.strip().splitlines():
                            pid, memory = line.split(",")
                            if int(pid.strip()) == os.getpid():
                                own += int(memory.strip())
                    f.write(json.dumps({"used_mib": current, "own_used_mib": own}) + "\n")
                    exceeded = (own >= 10240 or total - current < 1536) if shared else current >= total / 2 - 512
                    if exceeded:
                        print("VRAM guard: stopping own process at configured memory threshold", flush=True)
                        os._exit(75)
                except Exception as e:
                    print("VRAM monitoring failed; stopping own process:", e, flush=True)
                    os._exit(76)
                stop.wait(0.25)

    threading.Thread(target=watch, daemon=True).start()
    return stop


if __name__ == "__main__":
    assert shared_budget(46068, 33062) == 8192
    try:
        shared_budget(46068, 35000)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Should refuse shared pilot without headroom")
    assert allocator_budget(46068, 1517) == 18432
    assert allocator_budget(46068, 9000) == 11986
    try:
        allocator_budget(46068, 22000)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Should refuse allocation with insufficient headroom")
    print("VRAM budget checks passed")
