"""Run the audited bilevel U-Net runner with an in-memory decoded-slice cache.

All command-line arguments are passed unchanged to
scripts/run_bilevel_adversarial_unet.py. The repository files are not modified.
Each slice is decoded once, then copied from RAM on later reads. Random edits
and PGD are still generated fresh, so the cache changes I/O and not the data.
Only training slices are cached (about 2.05 GiB for the full fold), because the
runner already holds validation and test slices in memory.
"""
from __future__ import annotations

import atexit
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "dip-coma-cn"
RUNNER = REPO / "scripts" / "run_bilevel_adversarial_unet.py"
CUDA_SAFE_EVALUATOR = Path(__file__).resolve().with_name("run_unet_evaluation_cuda_safe.py")
sys.path.insert(0, str(REPO))

import torch

import src.supervised_localization as localization

_decode_image_tensor = localization._image_tensor
_slice_cache: dict[tuple[str, int], torch.Tensor] = {}
_train_root = (Path(sys.argv[sys.argv.index("--train") + 1]).resolve()
               if "--train" in sys.argv else (REPO / "data" / "cpaisd" / "train").resolve())


def cached_image_tensor(path, image_size):
    key = (str(path), int(image_size))
    if key in _slice_cache:
        return _slice_cache[key].clone()
    image = _decode_image_tensor(path, image_size)
    if Path(path).resolve().is_relative_to(_train_root):
        _slice_cache[key] = image
        return image.clone()
    return image


def report_resources():
    cached_bytes = sum(item.numel() * item.element_size() for item in _slice_cache.values())
    print(f"[cache] slices={len(_slice_cache)} ram_gib={cached_bytes / 1024 ** 3:.2f}", flush=True)
    if torch.cuda.is_available():
        print(f"[vram] training_process_peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} "
              f"peak_reserved_gib={torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}", flush=True)


import builtins
from datetime import datetime

_plain_print = builtins.print


def timestamped_print(*values, **kwargs):
    _plain_print(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), *values, **kwargs)


builtins.print = timestamped_print
localization._image_tensor = cached_image_tensor
atexit.register(report_resources)
if sys.platform == "win32":
    import ctypes
    # Block idle sleep only while this process runs; Windows power settings are unchanged.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
sys.argv = [str(RUNNER), *sys.argv[1:]]
runner_spec = importlib.util.spec_from_file_location("bilevel_adversarial_runner", RUNNER)
runner = importlib.util.module_from_spec(runner_spec)
sys.modules[runner_spec.name] = runner
runner_spec.loader.exec_module(runner)
runner.EVALUATOR = CUDA_SAFE_EVALUATOR
runner.main()
