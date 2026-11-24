"""Train the fixed-loss, width-24 control using the audited f895971 data and loss code."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "train", "validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from src import supervised_localization as loc

    args.output.mkdir(parents=True, exist_ok=False)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    paths = sorted(args.train.rglob("raw.dcm"))
    valpaths = sorted(args.validation.rglob("raw.dcm"))
    if (len(paths), len(valpaths)) != (8376, 980):
        raise ValueError("Expected 8376 training and 980 validation raw.dcm slices")
    validation = loc._load_paths(valpaths, 256)
    groups = sorted({item["study_group"] for item in validation})
    if len(validation) != 980 or len(groups) != 10:
        raise ValueError("Incomplete validation fold")
    selection = [r for r in validation if r["study_group"] in groups[5:]]
    rng = random.Random(42 + 65537)
    subset = [selection[i] for i in sorted(rng.sample(range(len(selection)), 256))]
    val = DataLoader(loc.SyntheticValidationDataset(subset, 42), batch_size=18)
    loc._image_tensor = lru_cache(maxsize=8376)(loc._image_tensor)
    dataset = loc.MixedAdversarialTrainingDataset(paths, 256, 42)
    loader = DataLoader(dataset, batch_size=6, shuffle=True,
                        generator=torch.Generator().manual_seed(42 + 7919))
    model = loc.build_localizer("unet", 24).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    config = dict(architecture="unet", base_channels=24, image_size=256, seed=42,
                  batch_size=18, train_slices=8376, epochs=30, adversarial_training=False,
                  loss_weighting="fixed", bce_weight=0.5, dice_weight=0.5, positive_weight=3,
                  optimizer="AdamW", lr=0.001, weight_decay=0.0001, scheduler=None,
                  copies_per_source="one clean, two copies of its unattacked synthetic edit",
                  selection="highest unattacked validation macro pixel IoU at 0.5")
    best = -1.0
    start = time.perf_counter()
    with (args.output / "training_history.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_loss", "validation_macro_iou", "elapsed_seconds"])
        for epoch in range(1, 31):
            dataset.set_epoch(epoch - 1)
            model.train()
            loss_sum = torch.zeros((), device="cuda")
            for clean, edit, target in loader:
                # Replace each PGD view with its source edit to match the batch balance and step count.
                x = torch.cat((clean, edit, edit)).cuda()
                y = torch.cat((torch.zeros_like(target), target, target)).cuda()
                opt.zero_grad(set_to_none=True)
                loss = loc._segmentation_loss(model(x), y, 3.0, 0.5)
                loss.backward()
                opt.step()
                loss_sum += loss.detach()
            score = loc._validation_localization_metrics(model, val, "cuda")["macro_pixel_iou"]
            elapsed = round(time.perf_counter() - start, 1)
            writer.writerow([epoch, loss_sum.item()/len(loader), score, elapsed])
            stream.flush()
            if score > best:
                best = score
                config.update(best_epoch=epoch, best_validation_macro_pixel_iou=best)
                tmp = args.output / "best.tmp"
                torch.save({"state_dict": model.state_dict(), "config": config}, tmp)
                tmp.replace(args.output / "unet_localizer.pt")
            print(f"Epoch {epoch}/30, validation IoU {score:.4f}, best {best:.4f}, elapsed {elapsed}s", flush=True)
    (args.output / "experiment_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
