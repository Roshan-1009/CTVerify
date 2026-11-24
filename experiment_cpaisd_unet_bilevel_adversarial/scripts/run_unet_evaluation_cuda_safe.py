"""Run the audited FGSM/PGD evaluator with CUDA attack outputs returned to CPU.

At commit f895971 the evaluator computes ``image - edited`` with the attacked
image on CUDA and the scripted edit on CPU, which raises a device mismatch.
Moving the attack output to CPU leaves the attack and metrics unchanged.
Repository files are not modified.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "dip-coma-cn" / "scripts" / "run_unet_fgsm_pgd_evaluation.py"

spec = importlib.util.spec_from_file_location("unet_fgsm_pgd_evaluation", EVALUATOR_PATH)
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)

audited_attack = evaluator._attack_evasion


def attack_on_cpu(model, image, epsilon, method, steps, step_size, seed, device):
    return audited_attack(model, image, epsilon, method, steps, step_size, seed, device).cpu()


evaluator._attack_evasion = attack_on_cpu

if __name__ == "__main__":
    evaluator.main()
