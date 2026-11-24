"""Print and save the three-model comparison from frozen calibration outputs."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "fair", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    names = {"retrained 30-epoch baseline": "Old baseline",
             "Fair Baseline": "Fair baseline",
             "upgraded bilevel adversarial U-Net": "Upgraded"}

    def read(name):
        frame = pd.concat([pd.read_csv(root / name) for root in (args.old, args.fair)])
        assert set(frame.model) == set(names)
        frame["model"] = frame.model.map(names)
        return frame

    iou = read("test_localization.csv").query("pixel_threshold_rule == 'validation_selected'")
    alarms = read("image_alarm_rules.csv").query("split == 'test' and rule == 'perturbed_q99'").copy()
    alarms["condition"] = alarms["set"]
    assert alarms.groupby("condition").images.nunique().eq(1).all()

    def table(frame, metric, scale, expected):
        values = frame.pivot(index="condition", columns="model", values=metric)[list(names.values())] * scale
        assert len(values) == expected and values.notna().all().all()
        values["Upgraded minus fair"] = values["Upgraded"] - values["Fair baseline"]
        return values.to_markdown(floatfmt=".5f")

    iou["condition"] = iou.family + "/" + iou.condition
    clean = alarms[alarms.condition.str.startswith("clean/")].copy()
    clean["study_fpr"] = clean.studies_alarmed / clean.studies
    edited = alarms[alarms.condition.str.startswith("edited/")].copy()
    pgd = edited[edited.condition.str.contains("/pgd10_")]
    report = [
        "# CPAISD controlled baseline (Ablation 2)",
        "The old baseline used width 8 and 512 training slices. The fair baseline and upgraded "
        "model used width 24 and all 8,376 slices. The fair baseline uses fixed equal BCE/Dice "
        "weights and ordinary synthetic edits. Differences measure the combined training change. "
        "The fair checkpoint was selected on unattacked validation IoU and the upgraded checkpoint on "
        "unattacked and PGD validation IoU. Each model's final thresholds were fitted on validation.",
        "## PGD recall on edited test slices (%)", table(pgd, "rate", 100, 6),
        "## All edited conditions: pixel IoU", table(iou, "pixel_iou", 1, 9),
        "## All edited conditions: slice recall (%)", table(edited, "rate", 100, 9),
        "## Clean-slice false alarms (%)", table(clean, "rate", 100, 5),
        "## Clean-study false alarms (%)", table(clean, "study_fpr", 100, 5),
        "Rate differences are percentage points. Each clean condition includes 809 slices in ten "
        "studies. Patch, sparse and held-out copy-move have 236, 152 and 139 edited test slices, respectively."
    ]
    result = "\n\n".join(report) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(result)
    print(result)


if __name__ == "__main__":
    main()
