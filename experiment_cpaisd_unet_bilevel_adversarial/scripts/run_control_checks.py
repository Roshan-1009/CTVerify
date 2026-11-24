"""Two control checks on the saved baseline and upgraded U-Nets.

Check 1, perturbed clean slices: every clean test slice (no edit) is perturbed
within the same L-inf budget (uniform noise, sign noise, and both PGD attacks).
If these slices raise alarms, the model reacts to perturbation, not tampering.

Check 2, common pixel threshold: the same 527 edited test images are scored
unattacked and under both PGD attacks at each checkpoint's own pixel threshold
and at a shared 0.5 threshold (the one used for validation checkpoint selection).

Test edits, selected studies, seeds, attacks and the clean-validation q99 image
threshold match the audited evaluator and run_max_logit_pgd.py. Repository files
are not modified. Results go to a fresh output directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

WRAPPERS = Path(__file__).resolve().parent
REPO = WRAPPERS.parent / "dip-coma-cn"
WORKSPACE = WRAPPERS.parents[1]
OUTPUTS = WORKSPACE / "outputs"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WRAPPERS))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_module("unet_fgsm_pgd_evaluation", REPO / "scripts" / "run_unet_fgsm_pgd_evaluation.py")
from run_max_logit_pgd import max_logit_pgd  # noqa: E402

MODELS = {
    "retrained 30-epoch baseline": OUTPUTS / "retrained_baseline_30ep" / "unet_localizer.pt",
    "upgraded bilevel adversarial U-Net":
        OUTPUTS / "bilevel_adversarial_w24_e30_b18" / "robust_training" / "unet_localizer.pt",
}
PRIOR = OUTPUTS / "final_comparison" / "final_model_comparison.csv"
LOG = None


def log(message):
    print(message, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = evaluator.build_localizer(config.get("architecture", "unet"),
                                      base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    model.requires_grad_(False)
    return model, config


@torch.no_grad()
def predict(model, image, device):
    logits = model(image.to(device).reshape(1, 1, *image.shape[-2:]))
    return float(evaluator._image_scores_from_logits(logits)[0]), torch.sigmoid(logits)[0, 0]


@torch.no_grad()
def image_threshold(model, records, device, quantile):
    scores = []
    for start in range(0, len(records), 8):
        batch = torch.stack([item["image"] for item in records[start:start + 8]])[:, None]
        scores.extend(evaluator._image_scores_from_logits(model(batch.to(device))).cpu().tolist())
    return float(np.percentile(scores, quantile))


def perturb(name, model, image, epsilon, steps, step_size, seed, device):
    source = image.to(device)
    if name == "none":
        return source
    if name == "uniform_noise":
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.empty_like(source).uniform_(-epsilon, epsilon, generator=generator)
        return (source + noise).clamp(0.0, 1.0)
    if name == "sign_noise":
        generator = torch.Generator(device=device).manual_seed(seed)
        sign = torch.randint(0, 2, source.shape, generator=generator, device=device).float() * 2 - 1
        return (source + epsilon * sign).clamp(0.0, 1.0)
    if name == "pgd10_mean_bce":
        return evaluator._attack_evasion(model, image, epsilon, "pgd", steps, step_size, seed, device)
    if name == "pgd10_max_logit":
        return max_logit_pgd(model, image, epsilon, steps, step_size, seed, device)
    raise ValueError(name)


def perturbed_clean_check(label, model, config, records, threshold, args, device):
    rows = []
    own = float(config["pixel_threshold"])
    conditions = ("none", "uniform_noise", "sign_noise", "pgd10_mean_bce", "pgd10_max_logit")
    for index, record in enumerate(records):
        for condition in conditions:
            image = perturb(condition, model, record["image"], args.epsilon, args.steps,
                            args.step_size, args.seed + index, device)
            score, probability = predict(model, image, device)
            rows.append({
                "model": label, "condition": condition, "slice_index": index,
                "study_group": record["study_group"], "image_score": score,
                "image_alarm": int(score >= threshold),
                "pixel_alarm_fraction_own": float((probability >= own).float().mean()),
                "pixel_alarm_fraction_0p5": float((probability >= 0.5).float().mean()),
                "linf": float((image.cpu() - record["image"]).abs().max()),
            })
        if (index + 1) % 200 == 0:
            log(f"  {label}: perturbed clean {index + 1}/{len(records)}")
    return rows


def edited_examples(records, seed, attack_fraction=0.20):
    study_ids = sorted({record["study_group"] for record in records})
    count = max(1, round(len(study_ids) * attack_fraction))
    for family_index, family in enumerate(evaluator.TEST_FAMILIES):
        rng = random.Random(seed + 1009 * (family_index + 1))
        selected = set(rng.sample(study_ids, min(count, len(study_ids))))
        examples = []
        for index, record in enumerate(records):
            if record["study_group"] not in selected:
                continue
            rng = random.Random(seed + index * 17 + 9001 * (family_index + 1))
            edited, truth = evaluator._apply_test_edit(record["image"], family, rng)
            examples.append((record, edited, truth))
        yield family, examples


def common_threshold_check(label, model, config, records, threshold, args, device):
    thresholds = {"own_checkpoint": float(config["pixel_threshold"]), "shared_0p5": 0.5}
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0, "alarms": 0})
    for family, examples in edited_examples(records, args.seed):
        for example_index, (_, edited, truth) in enumerate(examples):
            seed = args.seed + example_index
            images = {
                "unattacked": edited,
                "pgd10_mean_bce": evaluator._attack_evasion(
                    model, edited, args.epsilon, "pgd", args.steps, args.step_size, seed, device),
                "pgd10_max_logit": max_logit_pgd(
                    model, edited, args.epsilon, args.steps, args.step_size, seed, device),
            }
            truth_mask = truth.to(device=device, dtype=torch.bool)
            for condition, image in images.items():
                score, probability = predict(model, image, device)
                for name, value in thresholds.items():
                    item = stats[(family, condition, name, value)]
                    mask = probability >= value
                    item["tp"] += int(torch.count_nonzero(truth_mask & mask))
                    item["fp"] += int(torch.count_nonzero(~truth_mask & mask))
                    item["fn"] += int(torch.count_nonzero(truth_mask & ~mask))
                    item["images"] += 1
                    item["alarms"] += int(score >= threshold)
        log(f"  {label}: edited {family} done ({len(examples)} images)")
    rows = []
    for (family, condition, name, value), item in stats.items():
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        rows.append({
            "model": label, "family": family, "attack": condition,
            "pixel_threshold_rule": name, "pixel_threshold": value,
            "positive_images": item["images"], "detected_images": item["alarms"],
            "image_recall": item["alarms"] / max(1, item["images"]),
            "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
            "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
            "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "pixel_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "flagged_area_over_true_area": (tp + fp) / (tp + fn) if tp + fn else float("nan"),
        })
    return rows


def summarize_clean(frame):
    rows = []
    for (label, condition), part in frame.groupby(["model", "condition"], sort=False):
        study = part.groupby("study_group").image_alarm.max()
        rows.append({
            "model": label, "condition": condition, "clean_slices": len(part),
            "false_alarm_slices": int(part.image_alarm.sum()),
            "clean_slice_fpr": float(part.image_alarm.mean()),
            "false_alarm_studies": int(study.sum()), "clean_studies": len(study),
            "clean_study_fpr": float(study.mean()),
            "median_image_score": float(part.image_score.median()),
            "mean_pixel_alarm_fraction_own": float(part.pixel_alarm_fraction_own.mean()),
            "mean_pixel_alarm_fraction_0p5": float(part.pixel_alarm_fraction_0p5.mean()),
            "max_linf": float(part.linf.max()),
        })
    return pd.DataFrame(rows)


def consistency(clean_summary, localization):
    prior = pd.read_csv(PRIOR)
    checks = []
    for label in MODELS:
        old = prior[prior.model == label]
        new_clean = clean_summary[(clean_summary.model == label) & (clean_summary.condition == "none")].iloc[0]
        checks.append({"model": label, "check": "clean_slice_fpr",
                       "previous": float(old.clean_slice_fpr.iloc[0]), "now": new_clean.clean_slice_fpr})
        own = localization[(localization.model == label)
                           & (localization.pixel_threshold_rule == "own_checkpoint")]
        for _, row in own.iterrows():
            match = old[(old.family == row.family) & (old.attack == row.attack)]
            if match.empty:
                continue
            for column in ("image_recall", "pixel_iou"):
                checks.append({"model": label, "check": f"{row.family}/{row.attack}/{column}",
                               "previous": float(match[column].iloc[0]), "now": float(row[column])})
    frame = pd.DataFrame(checks)
    frame["abs_difference"] = (frame.now - frame.previous).abs()
    return frame


def main():
    global LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path,
                        default=Path(r"C:\Users\nidal\Research Project\dataset\dataset\val"))
    parser.add_argument("--test", type=Path, default=WORKSPACE / "work" / "cpaisd" / "dataset" / "test")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "control_checks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.004)
    parser.add_argument("--image-quantile", type=float, default=99.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    LOG = args.output / "control_checks.log"
    started = time.perf_counter()
    device = torch.device(args.device)

    log("Reading DICOM slices (header scan takes a few minutes)")
    first_config = torch.load(next(iter(MODELS.values())), map_location="cpu", weights_only=False)["config"]
    image_size = int(first_config["image_size"])
    validation = evaluator._load_paths(evaluator.find_dicom_files(args.validation), image_size)
    test = evaluator._load_paths(evaluator.find_dicom_files(args.test), image_size)
    log(f"validation slices={len(validation)} test slices={len(test)}")

    clean_rows, localization_rows, thresholds = [], [], {}
    for label, path in MODELS.items():
        model, config = load_model(path, device)
        if int(config["image_size"]) != image_size:
            raise RuntimeError("Checkpoints use different image sizes.")
        threshold = image_threshold(model, validation, device, args.image_quantile)
        thresholds[label] = {"image_threshold": threshold, "pixel_threshold": float(config["pixel_threshold"])}
        log(f"{label}: image threshold={threshold:.4f} pixel threshold={config['pixel_threshold']}")
        clean_rows += perturbed_clean_check(label, model, config, test, threshold, args, device)
        localization_rows += common_threshold_check(label, model, config, test, threshold, args, device)
        del model
        torch.cuda.empty_cache()

    clean = pd.DataFrame(clean_rows)
    if clean.linf.max() > args.epsilon + 1e-6:
        raise RuntimeError(f"Perturbation exceeded budget: {clean.linf.max()}")
    clean_summary = summarize_clean(clean)
    localization = pd.DataFrame(localization_rows).sort_values(
        ["family", "attack", "pixel_threshold_rule", "model"]).reset_index(drop=True)
    checks = consistency(clean_summary, localization)

    clean.to_csv(args.output / "perturbed_clean_per_slice.csv", index=False)
    clean_summary.to_csv(args.output / "perturbed_clean_summary.csv", index=False)
    localization.to_csv(args.output / "common_threshold_localization.csv", index=False)
    checks.to_csv(args.output / "consistency_with_previous_runs.csv", index=False)
    info = {
        "thresholds": thresholds, "seed": args.seed, "epsilon_linf_normalized": args.epsilon,
        "pgd_steps": args.steps, "pgd_step_size": args.step_size,
        "image_threshold_rule": f"clean validation q{args.image_quantile} of max pre-sigmoid logit",
        "max_consistency_difference": float(checks.abs_difference.max()),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        if torch.cuda.is_available() else None,
    }
    (args.output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    log(clean_summary.to_string(index=False))
    log(localization.to_string(index=False))
    log(checks.sort_values("abs_difference", ascending=False).head(8).to_string(index=False))
    log(json.dumps(info, indent=2))
    log("[complete]")


if __name__ == "__main__":
    main()
