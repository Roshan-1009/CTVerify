"""Evaluate a U-Net under L-inf PGD aimed directly at its max-logit image alarm.

The audited evaluator supplies the test edits, selected studies, clean-validation
q99 threshold, pixel threshold, and metrics. Only the PGD objective is replaced.
The evaluator's mean-BCE PGD is run in the same job as the reference condition.
Repository files are not modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1] / "dip-coma-cn"
EVALUATOR_PATH = REPO / "scripts" / "run_unet_fgsm_pgd_evaluation.py"
LINF_TOLERANCE = 1e-6


def load_evaluator():
    spec = importlib.util.spec_from_file_location("unet_fgsm_pgd_evaluation", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def max_logit_pgd(model, image, epsilon, steps, step_size, seed, device):
    """Minimize the highest logit and return the lowest-scoring iterate."""
    source = image.detach().to(device).unsqueeze(0).unsqueeze(0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    candidate = source + torch.empty_like(source).uniform_(-epsilon, epsilon, generator=generator)
    candidate = candidate.clamp(0.0, 1.0)
    best = candidate.detach().clone()
    best_score = float("inf")
    for _ in range(steps):
        candidate = candidate.detach().requires_grad_(True)
        score = model(candidate).flatten(start_dim=1).amax(dim=1).sum()
        score_value = float(score.detach())
        if score_value < best_score:
            best_score = score_value
            best = candidate.detach().clone()
        gradient = torch.autograd.grad(score, candidate, only_inputs=True)[0]
        candidate = candidate.detach() - step_size * gradient.sign()
        candidate = torch.maximum(torch.minimum(candidate, source + epsilon), source - epsilon)
        candidate = candidate.clamp(0.0, 1.0)
    with torch.no_grad():
        final_score = float(model(candidate).flatten(start_dim=1).amax(dim=1).sum())
    if final_score < best_score:
        best = candidate.detach().clone()
    return best[0, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.004)
    parser.add_argument("--max-images-per-family", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    evaluator = load_evaluator()
    reference_attack = evaluator._attack_evasion
    read_records = evaluator._load_paths
    record_cache = {}

    def cached_records(paths, image_size):
        key = (tuple(str(path) for path in paths), image_size)
        if key not in record_cache:
            record_cache[key] = read_records(paths, image_size)
        return record_cache[key]

    def reference_attack_on_cpu(model, image, epsilon, method, steps, step_size, seed, device):
        # The audited evaluator subtracts the CPU edit from the returned attack image.
        return reference_attack(model, image, epsilon, method, steps, step_size, seed, device).cpu()

    def targeted_attack(model, image, epsilon, method, steps, step_size, seed, device):
        if method != "pgd":
            return reference_attack_on_cpu(model, image, epsilon, method, steps, step_size, seed, device)
        return max_logit_pgd(model, image, epsilon, steps, step_size, seed, device).cpu()

    evaluator._load_paths = cached_records
    results = {}
    for objective, attack in (("mean_bce", reference_attack_on_cpu), ("max_logit", targeted_attack)):
        evaluator._attack_evasion = attack
        condition_dir = args.output / f"{objective}_pgd"
        print(f"Running {objective} PGD for {args.model_label}", flush=True)
        evaluator.run_evaluation(
            args.checkpoint, args.test, condition_dir, seed=args.seed,
            epsilon=args.epsilon, pgd_steps=args.steps, pgd_step_size=args.step_size,
            max_images_per_family=args.max_images_per_family, device_name=args.device,
            validation_dir=str(args.validation), attack_methods=("baseline", "pgd"),
        )
        per_image = pd.read_csv(condition_dir / "adversarial_comparison_per_image.csv")
        max_linf = float(per_image.loc[per_image["method"] == "pgd", "linf_from_scripted_edit"].max())
        if max_linf > args.epsilon + LINF_TOLERANCE:
            raise RuntimeError(f"{objective} PGD exceeded L-inf budget: {max_linf}")
        results[objective] = {
            "summary": pd.read_csv(condition_dir / "adversarial_comparison_summary.csv"),
            "config": json.loads((condition_dir / "attack_config.json").read_text(encoding="utf-8")),
            "max_linf": max_linf,
        }

    reference = results["mean_bce"]["summary"]
    targeted = results["max_logit"]["summary"]
    reference_clean = reference[reference["method"] == "baseline"].reset_index(drop=True)
    targeted_clean = targeted[targeted["method"] == "baseline"].reset_index(drop=True)
    for column in ("positive_images", "image_recall", "pixel_iou", "clean_test_file_fpr",
                   "clean_test_study_fpr"):
        if not np.allclose(reference_clean[column], targeted_clean[column]):
            raise RuntimeError(f"Unattacked results differ between runs: {column}")
    if results["mean_bce"]["config"]["image_threshold"] != results["max_logit"]["config"]["image_threshold"]:
        raise RuntimeError("Image thresholds differ between runs.")

    frames = [
        reference_clean.assign(attack="unattacked"),
        reference[reference["method"] == "pgd"].assign(attack="pgd10_mean_bce"),
        targeted[targeted["method"] == "pgd"].assign(attack="pgd10_max_logit"),
    ]
    comparison = pd.concat(frames, ignore_index=True)
    comparison.insert(0, "model", args.model_label)
    columns = ["model", "family", "attack", "positive_images", "detected_images", "image_recall",
               "pixel_iou", "attack_success_rate", "clean_test_file_fpr", "clean_test_study_fpr"]
    comparison = comparison[columns].sort_values(["family", "attack"]).reset_index(drop=True)
    comparison.to_csv(args.output / "max_logit_pgd_comparison.csv", index=False)
    run_info = {
        "checkpoint": str(args.checkpoint.resolve()), "model_label": args.model_label,
        "seed": args.seed, "epsilon_linf_normalized": args.epsilon, "steps": args.steps,
        "step_size": args.step_size, "random_start": True, "restarts": 1,
        "max_logit_returns_best_iterate": True,
        "image_threshold": results["max_logit"]["config"]["image_threshold"],
        "pixel_threshold": results["max_logit"]["config"]["pixel_threshold"],
        "max_linf_mean_bce": results["mean_bce"]["max_linf"],
        "max_linf_max_logit": results["max_logit"]["max_linf"],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_vram_allocated_gib": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                    if torch.cuda.is_available() else None),
    }
    (args.output / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(run_info, indent=2), flush=True)


if __name__ == "__main__":
    main()
