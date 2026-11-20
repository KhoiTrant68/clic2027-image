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
import time


def allocator_budget(total_mib, used_mib):
    # Leave room for CUDA context/libraries and for unrelated services.
    budget = min(18 * 1024, total_mib * 0.5 - used_mib - 2048)
    if budget < 1024:
        raise RuntimeError("Insufficient headroom under half-device VRAM limit")
    return int(budget)


def shared_budget(total_mib, used_mib, cap_mib=8 * 1024):
    # Opt-in shared-GPU mode: allocator cap (default 8 GiB), 2 GiB context room, 1.5 GiB free.
    if total_mib - used_mib < cap_mib + 2048 + 1536:
        raise RuntimeError(f"Insufficient free VRAM for the {cap_mib} MiB shared-GPU cap")
    return cap_mib


def install(log_path):
    import torch
    if os.environ.get("VRAM_GUARD") == "off":  # dedicated GPU (not lab09): no allocator cap, no monitor thread
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
        print("VRAM guard disabled (VRAM_GUARD=off)", flush=True)
        return threading.Event()
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical.isdigit():
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index")

    def read():
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", physical, "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=float(os.environ.get("VRAM_GUARD_TIMEOUT", 5)))
        return [int(v.strip()) for v in raw.strip().split(",")]

    shared = os.environ.get("AEIC_SHARED_SMALL") == "1"
    # SHARED_CAP_MIB raises the shared-mode allocator cap; the process limit stays cap + 2 GiB.
    cap = int(os.environ.get("SHARED_CAP_MIB", 8 * 1024))
    # Long training jobs on a busy host: nvidia-smi can stall for minutes; allow more misses before failing closed.
    max_failures = int(os.environ.get("VRAM_GUARD_MAX_FAILURES", 6))
    probe_timeout = float(os.environ.get("VRAM_GUARD_TIMEOUT", 5))
    # VRAM_GUARD_WAIT=seconds: if other users leave too little free memory, wait (re-check every minute) instead of
    # failing at start; queued jobs then resume by themselves when the device frees up.
    deadline = time.time() + float(os.environ.get("VRAM_GUARD_WAIT", 0))
    while True:
        total, used = read()
        try:
            budget = shared_budget(total, used, cap) if shared else allocator_budget(total, used)
            break
        except RuntimeError as e:
            if time.time() >= deadline:
                raise
            print(f"VRAM guard: {e} (used {used}/{total} MiB); waiting", flush=True)
            time.sleep(60)
    process_limit = budget + 2048
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.set_num_threads(2)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    state = {"total_mib": total, "initial_used_mib": used, "allocator_cap_mib": budget,
             "device_limit_mib": None if shared else total / 2, "physical_gpu": physical,
             "shared_small": shared, "process_limit_mib": process_limit if shared else None}
    path.with_suffix(".json").write_text(json.dumps(state, indent=2))
    print("VRAM guard", state, flush=True)

    def watch():
        failures = 0  # consecutive nvidia-smi failures; the allocator cap still bounds PyTorch meanwhile
        with path.open("a", buffering=1) as f:
            while not stop.is_set():
                try:
                    _, current = read()
                    own = 0
                    if shared:
                        processes = subprocess.check_output(["nvidia-smi", "-i", physical,
                            "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                            text=True, timeout=probe_timeout)
                        for line in processes.strip().splitlines():
                            pid, memory = line.split(",")
                            if int(pid.strip()) == os.getpid():
                                own += int(memory.strip())
                    f.write(json.dumps({"used_mib": current, "own_used_mib": own}) + "\n")
                    failures = 0
                    exceeded = (own >= process_limit or total - current < 1536) if shared else current >= total / 2 - 512
                    if exceeded:
                        print("VRAM guard: stopping own process at configured memory threshold", flush=True)
                        os._exit(75)
                except Exception as e:
                    failures += 1
                    if failures >= max_failures:  # default ~30 s without a reading: fail closed
                        print("VRAM monitoring failed repeatedly; stopping own process:", e, flush=True)
                        os._exit(76)
                    print(f"VRAM monitoring hiccup {failures}/{max_failures}:", e, flush=True)
                stop.wait(0.25)

    threading.Thread(target=watch, daemon=True).start()
    return stop


if __name__ == "__main__":
    assert shared_budget(46068, 33062) == 8192
    assert shared_budget(81559, 45514, 20480) == 20480
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
