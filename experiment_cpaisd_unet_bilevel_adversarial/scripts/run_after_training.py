"""Wait for full training, run max-logit PGD, and build the final comparison.

The job waits without importing PyTorch, so it uses little memory during training.
It runs the attack only if training exits with code 0.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUTS = WORKSPACE / "outputs"
TRAIN_NAME = "bilevel_adversarial_w24_e30_b18"
TRAIN_LOG = OUTPUTS / f"{TRAIN_NAME}.log"
TRAIN_DIR = OUTPUTS / TRAIN_NAME
CHECKPOINT = TRAIN_DIR / "robust_training" / "unet_localizer.pt"
STANDARD_CSV = TRAIN_DIR / "baseline_vs_bilevel_adversarial.csv"
BASELINE_ATTACK = OUTPUTS / "max_logit_pgd" / "retrained_baseline"
UPGRADED_ATTACK = OUTPUTS / "max_logit_pgd" / "upgraded_bilevel_adversarial"
FINAL_DIR = OUTPUTS / "final_comparison"
FOLLOWUP_LOG = OUTPUTS / "overnight_followup.log"
ATTACK_SCRIPT = Path(__file__).resolve().with_name("run_max_logit_pgd.py")
VALIDATION = Path(r"C:\Users\nidal\Research Project\dataset\dataset\val")
TEST = WORKSPACE / "work" / "cpaisd" / "dataset" / "test"
BASELINE_LABEL = "retrained 30-epoch baseline"
UPGRADED_LABEL = "upgraded bilevel adversarial U-Net"


def log(message):
    with FOLLOWUP_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def training_exit_code():
    if not TRAIN_LOG.exists():
        return None
    with TRAIN_LOG.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 4096))
        tail = handle.read().decode("utf-8", errors="replace")
    for line in reversed(tail.splitlines()):
        if line.startswith("[exit] code="):
            return int(line.split("=", 1)[1].strip())
    return None


def process_running(pid):
    result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                            capture_output=True, text=True, check=False)
    return str(pid) in result.stdout


def wait_for_training(pid):
    while True:
        code = training_exit_code()
        if code is not None:
            return code
        if not process_running(pid):
            time.sleep(60)
            code = training_exit_code()
            return code if code is not None else -1
        time.sleep(30)


def survives_both(attack_dir):
    import pandas as pd

    key = ["family", "file_id"]

    def alarms(name):
        frame = pd.read_csv(attack_dir / f"{name}_pgd" / "adversarial_comparison_per_image.csv")
        frame = frame[frame["method"] == "pgd"][key + ["image_alarm"]]
        return frame.rename(columns={"image_alarm": f"{name}_alarm"})

    joined = alarms("mean_bce").merge(alarms("max_logit"), on=key, validate="one_to_one")
    joined["detected"] = joined["mean_bce_alarm"] & joined["max_logit_alarm"]
    return joined.groupby("family", as_index=False).agg(
        positive_images=("file_id", "size"), detected_images=("detected", "sum"))


def model_rows(label, source_label, attack_dir, standard):
    import pandas as pd

    summary = pd.read_csv(attack_dir / "max_logit_pgd_comparison.csv")
    slice_fpr = float(summary["clean_test_file_fpr"].iloc[0])
    study_fpr = float(summary["clean_test_study_fpr"].iloc[0])
    columns = ["family", "attack", "positive_images", "detected_images", "image_recall", "pixel_iou"]
    fgsm = standard[(standard["model"] == source_label) & (standard["method"] == "fgsm")].copy()
    if fgsm.empty:
        raise ValueError(f"No FGSM rows for {source_label}")
    fgsm["attack"] = "fgsm"
    both = survives_both(attack_dir)
    both["attack"] = "detected_under_both_pgd"
    both["image_recall"] = both["detected_images"] / both["positive_images"]
    both["pixel_iou"] = float("nan")
    rows = pd.concat([summary[columns], fgsm[columns], both[columns]], ignore_index=True)
    rows.insert(0, "model", label)
    rows["pixel_dice"] = 2 * rows["pixel_iou"] / (1 + rows["pixel_iou"])
    rows["clean_slice_fpr"] = slice_fpr
    rows["clean_study_fpr"] = study_fpr
    return rows


def build_comparison(baseline_attack, upgraded_attack, standard_csv, output):
    import pandas as pd

    standard = pd.read_csv(standard_csv)
    final = pd.concat([
        model_rows(BASELINE_LABEL, "Baseline 30-epoch U-Net", baseline_attack, standard),
        model_rows(UPGRADED_LABEL, "Bilevel + adversarial U-Net", upgraded_attack, standard),
    ], ignore_index=True)
    order = {"unattacked": 0, "fgsm": 1, "pgd10_mean_bce": 2, "pgd10_max_logit": 3,
             "detected_under_both_pgd": 4}
    final["_order"] = final["attack"].map(order)
    final = final.sort_values(["family", "_order", "model"]).drop(columns="_order")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "final_model_comparison.csv"
    final.to_csv(path, index=False)
    return path, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pid", type=int)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--baseline-attack", type=Path, default=BASELINE_ATTACK)
    parser.add_argument("--upgraded-attack", type=Path, default=UPGRADED_ATTACK)
    parser.add_argument("--standard-csv", type=Path, default=STANDARD_CSV)
    parser.add_argument("--output", type=Path, default=FINAL_DIR)
    args = parser.parse_args()

    if args.compare_only:
        path, final = build_comparison(args.baseline_attack, args.upgraded_attack,
                                       args.standard_csv, args.output)
        print(final.to_string(index=False))
        print(f"Saved: {path}")
        return 0
    if args.train_pid is None:
        parser.error("--train-pid is required unless --compare-only is used")
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    log(f"[waiting] training pid={args.train_pid}")
    code = wait_for_training(args.train_pid)
    if code != 0 or not CHECKPOINT.is_file() or not STANDARD_CSV.is_file():
        log(f"[stopped] training code={code}; checkpoint={CHECKPOINT.is_file()}; "
            f"comparison={STANDARD_CSV.is_file()}")
        return 1
    log("[attack] starting max-logit PGD on upgraded checkpoint")
    command = [sys.executable, "-u", str(ATTACK_SCRIPT), "--checkpoint", str(CHECKPOINT),
               "--validation", str(VALIDATION), "--test", str(TEST),
               "--output", str(UPGRADED_ATTACK), "--model-label", UPGRADED_LABEL,
               "--seed", "42", "--epsilon", "0.02", "--steps", "10", "--step-size", "0.004",
               "--max-images-per-family", "0", "--device", "cuda"]
    with FOLLOWUP_LOG.open("a", encoding="utf-8") as handle:
        attack = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if attack.returncode != 0:
        log(f"[stopped] max-logit attack code={attack.returncode}")
        return attack.returncode
    path, _ = build_comparison(BASELINE_ATTACK, UPGRADED_ATTACK, STANDARD_CSV, FINAL_DIR)
    log(f"[complete] final comparison: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
