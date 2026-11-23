"""Cryptographic-first single-file and directory verification workflows."""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import (STUDY_MANIFEST_DIR, STUDY_MANIFEST_SUFFIX, compare_study_instances,
                       replay_transaction, verify_instance, verify_study_envelope)


def verify_then_analyze(dicom_path: str | Path, manifest_path: str | Path,
                        trust_registry_path: str | Path, *, if_model_path: str | Path | None = None,
                        autoencoder_model_path: str | Path | None = None,
                        pixel_model_path: str | Path | None = None,
                        unet_checkpoint_path: str | Path | None = None,
                        run_ml_on_unverified: bool = False, expected_signer: str | None = None,
                        receiver_id: str | None = None,
                        replay_cache_path: str | Path | None = None) -> dict:
    """Return separate cryptographic, rules, semantic, IF, and AE results.

    ML is skipped after a cryptographic failure by default. All analysis uses
    the dataset parsed from the exact bytes that were verified. U-Net and patch
    pixel inference are invoked only when their saved model paths are supplied.
    """
    registry = json.loads(Path(trust_registry_path).read_text(encoding="utf-8"))
    verified = verify_instance(dicom_path, manifest_path, registry, expected_signer=expected_signer,
                               receiver_id=receiver_id, replay_cache_path=replay_cache_path)
    crypto = verified.result
    result = {"cryptographic": crypto, "structural": None, "semantic": None,
              "metadata_anomaly": {}, "pixel_anomaly": {"status": "NOT_CONFIGURED"},
              "segmentation": {"status": "NOT_CONFIGURED"}}
    if crypto["status"] != "VERIFIED" and not run_ml_on_unverified:
        result["ml_status"] = "SKIPPED_CRYPTOGRAPHIC_POLICY"
        return result
    if verified.dataset is None:
        result["ml_status"] = "SKIPPED_UNREADABLE_DICOM"
        return result

    from ..validation import validate_datasets
    from ..semantic_consistency import validate_semantic_consistency
    from ..pixel_metadata_consistency import check_pixel_metadata_consistency
    from ..anomaly import load_model, predict_anomalies
    from ..deep_anomaly import predict_autoencoder_with_explanations

    ds = verified.dataset
    key = Path(dicom_path).name
    items = [(key, ds)]
    result["structural"] = validate_datasets(items)[key]
    result["semantic"] = {"status": "REQUIRES_STUDY_CONTEXT",
                           "reason": "single-file analysis cannot assess peer-slice agreement"}
    result["cross_layer"] = check_pixel_metadata_consistency(ds)

    if if_model_path and Path(if_model_path).is_file():
        try:
            bundle = load_model(str(if_model_path))
            flag, score = predict_anomalies(items, bundle)[key]
            result["metadata_anomaly"]["isolation_forest"] = {"status": "COMPLETED", "flag": int(flag), "score": float(score),
                "threshold": float(bundle.get("threshold", 0.0))}
        except Exception as exc:
            result["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_UNAVAILABLE"}
    if autoencoder_model_path and Path(autoencoder_model_path).is_file():
        try:
            flag, score, features = predict_autoencoder_with_explanations(items, str(autoencoder_model_path))[key]
            result["metadata_anomaly"]["autoencoder"] = {"status": "COMPLETED", "flag": int(flag), "reconstruction_error": float(score),
                                                             "top_features": features}
        except Exception as exc:
            result["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_UNAVAILABLE"}

    import numpy as np
    import torch
    import torch.nn.functional as F
    from ..no_reference_pixel import _normalized_image, _patch_feature_grid, _mask_from_patch_scores

    def mask_summary(mask):
        flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
        counts, current, run = [], 0, 0
        for value in flat:
            if int(value) == current:
                run += 1
            else:
                counts.append(run); run = 1; current = int(value)
        counts.append(run)
        return {"shape": list(np.asarray(mask).shape), "positive_pixels": int(flat.sum()), "rle_counts": counts}

    if pixel_model_path and Path(pixel_model_path).is_file():
        try:
            import joblib
            bundle = joblib.load(pixel_model_path)
            image = _normalized_image(ds)
            features, grid_shape, image_shape = _patch_feature_grid(image, int(bundle.get("patch_size", 16)))
            scores = bundle["model"].score_samples(features)
            threshold = float(bundle["patch_threshold"])
            mask = _mask_from_patch_scores(scores, grid_shape, image_shape, threshold)
            image_score = float(np.min(scores))
            result["pixel_anomaly"] = {"status": "COMPLETED", "flag": int(image_score < float(bundle["file_threshold"])),
                                        "score": image_score, "image_threshold": float(bundle["file_threshold"]),
                                        "mask": mask_summary(mask)}
        except Exception as exc:
            result["pixel_anomaly"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["pixel_anomaly"] = {"status": "MODEL_UNAVAILABLE"}

    if unet_checkpoint_path and Path(unet_checkpoint_path).is_file():
        try:
            from ..supervised_localization import AttentionUNetLocalizer, UNetLocalizer
            checkpoint = torch.load(unet_checkpoint_path, map_location="cpu", weights_only=True)
            config = checkpoint.get("config", {})
            architecture = config.get("architecture", "unet")
            model_class = AttentionUNetLocalizer if architecture == "attention_unet" else UNetLocalizer
            model = model_class(base_channels=int(config.get("base_channels", 8)))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            image = _normalized_image(ds)
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
            tensor = F.interpolate(tensor, size=(int(config.get("image_size", 256)),) * 2,
                                   mode="bilinear", align_corners=False)
            with torch.no_grad():
                probability = torch.sigmoid(model(tensor))[0, 0].numpy()
            pixel_threshold = float(config.get("pixel_threshold", 0.5))
            image_threshold = float(config.get("image_threshold", 1.0))
            mask = probability >= pixel_threshold
            result["segmentation"] = {"status": "COMPLETED", "flag": int(float(probability.max()) >= image_threshold),
                                      "architecture": architecture,
                                      "max_probability": float(probability.max()), "image_threshold": image_threshold,
                                      "pixel_threshold": pixel_threshold, "mask": mask_summary(mask),
                                      "model_input_size": list(probability.shape)}
        except Exception as exc:
            result["segmentation"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["segmentation"] = {"status": "MODEL_UNAVAILABLE"}
    result["ml_status"] = "COMPLETED"
    return result


def verify_directory_then_analyze(dicom_root: str | Path, manifest_root: str | Path,
                                  trust_registry_path: str | Path, *,
                                  if_model_path: str | Path | None = None,
                                  autoencoder_model_path: str | Path | None = None,
                                  pixel_model_path: str | Path | None = None,
                                  unet_checkpoint_path: str | Path | None = None,
                                  expected_signer: str | None = None, receiver_id: str | None = None,
                                  replay_cache_path: str | Path | None = None) -> dict:
    """Verify per-file sidecars and study manifests, then analyze verified objects.

    Manifest sidecars are resolved as ``manifest_root / (relative_path +
    '.manifest.json')``. Signed study manifests are read from
    ``manifest_root / '_studies'``. ``overall_status`` is ``VERIFIED`` only when
    every discovered file verifies and every study matches a signed, complete
    instance list. Public UIDs and source paths are not copied into the returned
    records; each result is keyed by a truncated hash of its relative path.
    Group consistency checks operate only on cryptographically verified files.
    """
    import hashlib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ..dicom_io import find_dicom_files, relative_key
    from ..validation import validate_datasets
    from ..semantic_consistency import validate_semantic_consistency
    from ..pixel_metadata_consistency import check_pixel_metadata_consistency
    from ..no_reference_pixel import _normalized_image, _patch_feature_grid, _mask_from_patch_scores

    root, manifest_root = Path(dicom_root).resolve(), Path(manifest_root).resolve()
    registry = json.loads(Path(trust_registry_path).read_text(encoding="utf-8"))
    paths = find_dicom_files(root)
    if not paths:
        raise ValueError(f"No readable DICOM files found under {root}")
    records, loaded = {}, {}
    for path in paths:
        rel = relative_key(path, root)
        token = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:20]
        sidecar = manifest_root / (rel + ".manifest.json")
        # Replay state is committed below, only after the whole directory is checked.
        verified = verify_instance(path, sidecar, registry, expected_signer=expected_signer, receiver_id=receiver_id)
        record = {"cryptographic": verified.result, "structural": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "semantic": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "cross_layer": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "metadata_anomaly": {}, "pixel_anomaly": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "segmentation": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"}}
        records[token] = record
        if verified.result["status"] == "VERIFIED":
            loaded[token] = verified

    study_token = lambda uid: hashlib.sha256(uid.encode("utf-8")).hexdigest()[:20]
    facts_by_study: dict[str, list[dict]] = {}
    for verified in loaded.values():
        facts_by_study.setdefault(verified.facts["study_instance_uid"], []).append(verified.facts)
    study_manifests: dict[str, list[dict]] = {}
    invalid_study_manifests = 0
    for manifest_path in sorted((manifest_root / STUDY_MANIFEST_DIR).glob(f"*{STUDY_MANIFEST_SUFFIX}")):
        manifest, failure = verify_study_envelope(manifest_path, registry, expected_signer=expected_signer,
                                                  receiver_id=receiver_id)
        if failure is not None:
            invalid_study_manifests += 1
        else:
            study_manifests.setdefault(manifest["study_instance_uid"], []).append(manifest)
    study_results: dict[str, dict] = {}
    study_entries: dict[str, dict] = {}
    for study_uid in sorted(set(facts_by_study) | set(study_manifests)):
        candidates = study_manifests.get(study_uid, [])
        if not candidates:
            outcome = {"status": "MISSING_REFERENCE", "details": {"reason": "no valid signed study manifest"}}
        elif len(candidates) > 1:
            outcome = {"status": "POLICY_REVIEW_REQUIRED", "details": {"reason": "multiple study manifests"}}
        else:
            outcome = compare_study_instances(candidates[0], facts_by_study.get(study_uid, []))
            study_entries[study_uid] = {"manifest_id": candidates[0]["manifest_id"], "object_key": f"study:{study_uid}",
                                        "object_version": candidates[0]["object_version"]}
        study_results[study_token(study_uid)] = outcome

    def directory_verified() -> bool:
        return (bool(paths) and len(loaded) == len(paths) and invalid_study_manifests == 0
                and bool(study_results) and all(item["status"] == "VERIFIED" for item in study_results.values()))

    replay_error = False
    if replay_cache_path is not None and (loaded or study_entries):
        entries = [verified.facts["replay_entry"] for verified in loaded.values()] + list(study_entries.values())
        try:
            conflicts = replay_transaction(replay_cache_path, entries, commit=directory_verified())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            replay_error = True
            conflicts = {entry["manifest_id"]: "replay cache is unreadable or unavailable" for entry in entries}
        for token, verified in list(loaded.items()):
            manifest_id = verified.facts["replay_entry"]["manifest_id"]
            if manifest_id in conflicts:
                records[token]["cryptographic"] = {"status": "POLICY_REVIEW_REQUIRED",
                                                   "details": {"reason": conflicts[manifest_id], "manifest_id": manifest_id}}
                del loaded[token]
        for study_uid, entry in study_entries.items():
            if entry["manifest_id"] in conflicts:
                study_results[study_token(study_uid)] = {"status": "POLICY_REVIEW_REQUIRED",
                                                         "details": {"reason": conflicts[entry["manifest_id"]]}}

    verified_items = [(token, verified.dataset, None) for token, verified in loaded.items()]
    overall_status = "VERIFIED" if directory_verified() and not replay_error else "NOT_VERIFIED"
    summary = {"overall_status": overall_status, "study_verification": study_results,
               "invalid_study_manifests": invalid_study_manifests,
               "protections": {"replay_cache": replay_cache_path is not None,
                               "expected_signer": expected_signer is not None,
                               "recipient_binding": receiver_id is not None}}

    if not verified_items:
        return {"files_discovered": len(paths), "files_verified": 0, "files_not_verified": len(paths),
                **summary, "results": records, "ml_status": "SKIPPED_NO_VERIFIED_FILES"}

    datasets = [(token, ds) for token, ds, _ in verified_items]
    structural = validate_datasets(datasets)
    semantic = validate_semantic_consistency(datasets)
    for token, ds, _ in verified_items:
        records[token]["structural"] = structural[token]
        records[token]["semantic"] = semantic[token]
        records[token]["cross_layer"] = check_pixel_metadata_consistency(ds)

    if if_model_path and Path(if_model_path).is_file():
        try:
            from ..anomaly import load_model, predict_anomalies
            bundle = load_model(str(if_model_path))
            predictions = predict_anomalies(datasets, bundle)
            for token, _ in datasets:
                flag, score = predictions[token]
                records[token]["metadata_anomaly"]["isolation_forest"] = {
                    "status": "COMPLETED", "flag": int(flag), "score": float(score),
                    "threshold": float(bundle.get("threshold", 0.0))}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_UNAVAILABLE"}

    if autoencoder_model_path and Path(autoencoder_model_path).is_file():
        try:
            from ..deep_anomaly import predict_autoencoder_with_explanations
            predictions = predict_autoencoder_with_explanations(datasets, str(autoencoder_model_path))
            for token, _ in datasets:
                flag, score, features = predictions[token]
                records[token]["metadata_anomaly"]["autoencoder"] = {
                    "status": "COMPLETED", "flag": int(flag), "reconstruction_error": float(score),
                    "top_features": features}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_UNAVAILABLE"}

    def mask_summary(mask):
        flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
        counts, current, run = [], 0, 0
        for value in flat:
            if int(value) == current:
                run += 1
            else:
                counts.append(run); run = 1; current = int(value)
        counts.append(run)
        return {"shape": list(np.asarray(mask).shape), "positive_pixels": int(flat.sum()), "rle_counts": counts}

    if pixel_model_path and Path(pixel_model_path).is_file():
        try:
            import joblib
            bundle = joblib.load(pixel_model_path)
            for token, ds, _ in verified_items:
                image = _normalized_image(ds)
                features, grid, shape = _patch_feature_grid(image, int(bundle.get("patch_size", 16)))
                scores = bundle["model"].score_samples(features)
                mask = _mask_from_patch_scores(scores, grid, shape, float(bundle["patch_threshold"]))
                image_score = float(np.min(scores))
                records[token]["pixel_anomaly"] = {"status": "COMPLETED",
                    "flag": int(image_score < float(bundle["file_threshold"])), "score": image_score,
                    "image_threshold": float(bundle["file_threshold"]), "mask": mask_summary(mask)}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["pixel_anomaly"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["pixel_anomaly"] = {"status": "MODEL_UNAVAILABLE"}

    if unet_checkpoint_path and Path(unet_checkpoint_path).is_file():
        try:
            from ..supervised_localization import AttentionUNetLocalizer, UNetLocalizer
            checkpoint = torch.load(unet_checkpoint_path, map_location="cpu", weights_only=True)
            config = checkpoint.get("config", {})
            architecture = config.get("architecture", "unet")
            model_class = AttentionUNetLocalizer if architecture == "attention_unet" else UNetLocalizer
            model = model_class(base_channels=int(config.get("base_channels", 8)))
            model.load_state_dict(checkpoint["state_dict"]); model.eval()
            image_size = int(config.get("image_size", 256))
            pixel_threshold = float(config.get("pixel_threshold", 0.5))
            image_threshold = float(config.get("image_threshold", 1.0))
            for token, ds, _ in verified_items:
                image = _normalized_image(ds)
                tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
                tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    probability = torch.sigmoid(model(tensor))[0, 0].numpy()
                mask = probability >= pixel_threshold
                records[token]["segmentation"] = {"status": "COMPLETED", "architecture": architecture,
                    "flag": int(float(probability.max()) >= image_threshold),
                    "max_probability": float(probability.max()), "image_threshold": image_threshold,
                    "pixel_threshold": pixel_threshold, "mask": mask_summary(mask),
                    "model_input_size": list(probability.shape)}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["segmentation"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["segmentation"] = {"status": "MODEL_UNAVAILABLE"}

    return {"files_discovered": len(paths), "files_verified": len(verified_items),
            "files_not_verified": len(paths) - len(verified_items), **summary, "results": records,
            "ml_status": "COMPLETED"}
