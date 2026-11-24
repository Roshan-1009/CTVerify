"""Choose pixel and alarm thresholds on the validation fold, then apply them to test.

Rules fixed before running:
- Pixel threshold: the grid value with the highest validation macro pixel IoU over
  the training edit families (patch, sparse) x {unattacked, PGD mean-BCE, PGD
  max-logit}. Copy-move is not used for selection, so it stays unseen.
- Image alarm threshold, three rules compared:
  clean_q99      q99 of clean validation max-logit (the evaluator's rule);
  perturbed_q99  the highest per-condition q99 over clean validation slices
                 unperturbed, with uniform noise, sign noise, and both PGD attacks;
  logit_zero     fixed at logit 0 (any pixel at probability 0.5).
Test edits, studies, seeds and attacks match run_control_checks.py and the
audited evaluator. Validation edits use the same generator on the validation
fold. Repository files are not modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

WRAPPERS = Path(__file__).resolve().parent
sys.path.insert(0, str(WRAPPERS))
import run_control_checks as cc  # noqa: E402  (loads the audited evaluator)
from run_max_logit_pgd import max_logit_pgd  # noqa: E402
from src.supervised_localization import TRAIN_FAMILIES  # noqa: E402

evaluator = cc.evaluator
OUTPUTS = cc.OUTPUTS
GRID = [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CLEAN_CONDITIONS = ("none", "uniform_noise", "sign_noise", "pgd10_mean_bce", "pgd10_max_logit")
EDIT_CONDITIONS = ("unattacked", "pgd10_mean_bce", "pgd10_max_logit")
LOG = None


def log(message):
    print(message, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def clean_rows(split, label, model, records, args, device, grid):
    rows = []
    for index, record in enumerate(records):
        for condition in CLEAN_CONDITIONS:
            image = cc.perturb(condition, model, record["image"], args.epsilon, args.steps,
                               args.step_size, args.seed + index, device)
            score, probability = cc.predict(model, image, device)
            fractions = (probability.flatten()[None, :] >= grid[:, None]).float().mean(1).cpu().tolist()
            row = {"split": split, "model": label, "condition": condition, "slice_index": index,
                   "study_group": record["study_group"], "image_score": score,
                   "linf": float((image.cpu() - record["image"]).abs().max())}
            row.update({f"pixel_alarm_fraction@{value:g}": fraction for value, fraction in zip(GRID, fractions)})
            rows.append(row)
        if (index + 1) % 250 == 0:
            log(f"  {label} {split}: clean {index + 1}/{len(records)}")
    return rows


def edited_rows(split, label, model, records, args, device, grid):
    rows = []
    for family, examples in cc.edited_examples(records, args.seed):
        for example_index, (record, edited, truth) in enumerate(examples):
            seed = args.seed + example_index
            images = {
                "unattacked": edited,
                "pgd10_mean_bce": evaluator._attack_evasion(
                    model, edited, args.epsilon, "pgd", args.steps, args.step_size, seed, device),
                "pgd10_max_logit": max_logit_pgd(
                    model, edited, args.epsilon, args.steps, args.step_size, seed, device),
            }
            truth_mask = truth.to(device=device, dtype=torch.bool).flatten()[None, :]
            for condition, image in images.items():
                score, probability = cc.predict(model, image, device)
                masks = probability.flatten()[None, :] >= grid[:, None]
                tp = (masks & truth_mask).sum(1).tolist()
                fp = (masks & ~truth_mask).sum(1).tolist()
                fn = (~masks & truth_mask).sum(1).tolist()
                linf = float((image.cpu() - edited).abs().max())
                for value, a, b, c in zip(GRID, tp, fp, fn):
                    rows.append({"split": split, "model": label, "family": family,
                                 "condition": condition, "image_index": example_index,
                                 "study_group": record["study_group"], "image_score": score,
                                 "linf": linf, "pixel_threshold": value, "tp": a, "fp": b, "fn": c})
        log(f"  {label} {split}: edited {family} done ({len(examples)} images)")
    return rows


def pixel_metrics(frame, keys):
    grouped = frame.groupby(keys, as_index=False)[["tp", "fp", "fn"]].sum()
    total = grouped.tp + grouped.fp + grouped.fn
    grouped["pixel_iou"] = np.where(total > 0, grouped.tp / total.where(total > 0, 1), 0.0)
    dice_total = 2 * grouped.tp + grouped.fp + grouped.fn
    grouped["pixel_dice"] = np.where(dice_total > 0, 2 * grouped.tp / dice_total.where(dice_total > 0, 1), 0.0)
    true_area = grouped.tp + grouped.fn
    grouped["flagged_area_over_true_area"] = (grouped.tp + grouped.fp) / true_area.where(true_area > 0)
    return grouped


def main():
    global LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path,
                        default=Path(r"C:\Users\nidal\Research Project\dataset\dataset\val"))
    parser.add_argument("--test", type=Path, default=cc.WORKSPACE / "work" / "cpaisd" / "dataset" / "test")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "validation_calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.004)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    LOG = args.output / "validation_calibration.log"
    started = time.perf_counter()
    device = torch.device(args.device)
    grid = torch.tensor(GRID, device=device)

    log("Reading DICOM slices (header scan takes a few minutes)")
    first_config = torch.load(next(iter(cc.MODELS.values())), map_location="cpu", weights_only=False)["config"]
    image_size = int(first_config["image_size"])
    splits = {
        "validation": evaluator._load_paths(evaluator.find_dicom_files(args.validation), image_size),
        "test": evaluator._load_paths(evaluator.find_dicom_files(args.test), image_size),
    }
    log(f"validation slices={len(splits['validation'])} test slices={len(splits['test'])}")

    clean, edited = [], []
    for label, path in cc.MODELS.items():
        model, config = cc.load_model(path, device)
        for split, records in splits.items():
            clean += clean_rows(split, label, model, records, args, device, grid)
            edited += edited_rows(split, label, model, records, args, device, grid)
        del model
        torch.cuda.empty_cache()
    clean = pd.DataFrame(clean)
    edited = pd.DataFrame(edited)
    worst = max(clean.linf.max(), edited.linf.max())
    if worst > args.epsilon + 1e-6:
        raise RuntimeError(f"Perturbation exceeded budget: {worst}")
    clean.to_csv(args.output / "clean_slice_scores.csv", index=False)
    edited.to_csv(args.output / "edited_pixel_counts.csv", index=False)

    # Pixel threshold: validation, training families only.
    validation_edits = edited[(edited.split == "validation") & edited.family.isin(TRAIN_FAMILIES)]
    selection = pixel_metrics(validation_edits, ["model", "pixel_threshold", "family", "condition"])
    selection = selection.groupby(["model", "pixel_threshold"], as_index=False).pixel_iou.mean()
    selection = selection.rename(columns={"pixel_iou": "validation_macro_pixel_iou"})
    selection.to_csv(args.output / "pixel_threshold_selection.csv", index=False)
    chosen = selection.loc[selection.groupby("model").validation_macro_pixel_iou.idxmax()].set_index("model")

    localization = pixel_metrics(edited, ["split", "model", "family", "condition", "pixel_threshold"])
    rows = []
    for label in cc.MODELS:
        for rule, value in (("validation_selected", float(chosen.loc[label, "pixel_threshold"])), ("fixed_0p5", 0.5)):
            part = localization[(localization.model == label) & np.isclose(localization.pixel_threshold, value)]
            rows.append(part.assign(pixel_threshold_rule=rule))
    test_localization = pd.concat(rows)
    test_localization = test_localization[test_localization.split == "test"].sort_values(
        ["family", "condition", "pixel_threshold_rule", "model"])
    test_localization.to_csv(args.output / "test_localization.csv", index=False)

    # Image alarm thresholds: validation clean slices only.
    thresholds = {}
    for label in cc.MODELS:
        part = clean[(clean.split == "validation") & (clean.model == label)]
        per_condition = part.groupby("condition").image_score.apply(lambda s: float(np.percentile(s, 99)))
        thresholds[label] = {"clean_q99": float(per_condition["none"]),
                             "perturbed_q99": float(per_condition.max()),
                             "perturbed_q99_set_by": str(per_condition.idxmax()),
                             "logit_zero": 0.0}
    alarm_rows = []
    one_threshold = edited[edited.pixel_threshold == GRID[0]]
    for label in cc.MODELS:
        for rule in ("clean_q99", "perturbed_q99", "logit_zero"):
            value = thresholds[label][rule]
            for split in ("validation", "test"):
                part = clean[(clean.split == split) & (clean.model == label)]
                for condition, group in part.groupby("condition", sort=False):
                    studies = (group.image_score >= value).groupby(group.study_group).max()
                    alarm_rows.append({"model": label, "rule": rule, "threshold": value, "split": split,
                                       "set": f"clean/{condition}", "images": len(group),
                                       "alarms": int((group.image_score >= value).sum()),
                                       "rate": float((group.image_score >= value).mean()),
                                       "studies_alarmed": int(studies.sum()), "studies": len(studies)})
                part = one_threshold[(one_threshold.split == split) & (one_threshold.model == label)]
                for (family, condition), group in part.groupby(["family", "condition"], sort=False):
                    alarm_rows.append({"model": label, "rule": rule, "threshold": value, "split": split,
                                       "set": f"edited/{family}/{condition}", "images": len(group),
                                       "alarms": int((group.image_score >= value).sum()),
                                       "rate": float((group.image_score >= value).mean()),
                                       "studies_alarmed": None, "studies": None})
    alarms = pd.DataFrame(alarm_rows)
    alarms.to_csv(args.output / "image_alarm_rules.csv", index=False)

    # Consistency with earlier runs on the test fold.
    previous_clean = pd.read_csv(OUTPUTS / "control_checks" / "perturbed_clean_summary.csv")
    previous_local = pd.read_csv(OUTPUTS / "control_checks" / "common_threshold_localization.csv")
    checks = []
    for label in cc.MODELS:
        for condition in CLEAN_CONDITIONS:
            old = previous_clean[(previous_clean.model == label) & (previous_clean.condition == condition)]
            new = alarms[(alarms.model == label) & (alarms.rule == "clean_q99") & (alarms.split == "test")
                         & (alarms.set == f"clean/{condition}")]
            checks.append({"model": label, "check": f"test clean/{condition} slice FPR (clean_q99)",
                           "previous": float(old.clean_slice_fpr.iloc[0]), "now": float(new.rate.iloc[0])})
        fixed = localization[(localization.model == label) & (localization.split == "test")
                             & np.isclose(localization.pixel_threshold, 0.5)]
        for _, row in fixed.iterrows():
            old = previous_local[(previous_local.model == label) & (previous_local.family == row.family)
                                 & (previous_local.attack == row.condition)
                                 & (previous_local.pixel_threshold_rule == "shared_0p5")]
            checks.append({"model": label, "check": f"test {row.family}/{row.condition} IoU@0.5",
                           "previous": float(old.pixel_iou.iloc[0]), "now": float(row.pixel_iou)})
    checks = pd.DataFrame(checks)
    checks["abs_difference"] = (checks.now - checks.previous).abs()
    checks.to_csv(args.output / "consistency_with_control_checks.csv", index=False)

    info = {
        "selected_pixel_threshold": {label: float(chosen.loc[label, "pixel_threshold"]) for label in cc.MODELS},
        "selected_validation_macro_pixel_iou": {label: float(chosen.loc[label, "validation_macro_pixel_iou"])
                                                for label in cc.MODELS},
        "image_thresholds": thresholds,
        "validation_edited_images": int(one_threshold[one_threshold.split == "validation"]
                                        .groupby("model").size().iloc[0] / len(EDIT_CONDITIONS)),
        "max_consistency_difference": float(checks.abs_difference.max()),
        "seed": args.seed, "epsilon_linf_normalized": args.epsilon,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        if torch.cuda.is_available() else None,
    }
    (args.output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    log(selection.round(4).to_string(index=False))
    log(test_localization[["model", "family", "condition", "pixel_threshold_rule", "pixel_threshold",
                           "pixel_iou", "pixel_dice", "flagged_area_over_true_area"]].round(4).to_string(index=False))
    log(alarms.round(4).to_string(index=False))
    log(checks.sort_values("abs_difference", ascending=False).head(8).to_string(index=False))
    log(json.dumps(info, indent=2))
    log("[complete]")


if __name__ == "__main__":
    main()
