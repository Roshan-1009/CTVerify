"""Exercise signed-manifest security cases on a copy-only DICOM test set.

No training occurs. Temporary experiment keys are deleted when the run exits.
Example: python scripts/run_crypto_security_experiment.py --input data/cpaisd/test --output reports/crypto_security
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import tempfile
import time
from pathlib import Path
import sys

import pydicom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dicom_io import find_dicom_files
from src.security.canonical import canonical_json_bytes, metadata_sha256, pixel_data_sha256, raw_file_sha256
from src.security.manifest import create_signed_manifest, save_signed_manifest, verify_signed_manifest
from src.security.privacy import deidentify_research_copy
from src.security.signing import (add_trusted_signer, generate_ed25519_keypair,
                                  generate_hmac_key, hmac_sha256, load_private_key,
                                  trusted_public_key, verify_bytes, verify_hmac_sha256)


def _write_ds(source: Path, destination: Path, change: str) -> int | None:
    ds = pydicom.dcmread(str(source), force=False)
    changed_pixel_count = None
    if change == "metadata":
        ds.PatientID = "TAMPERED-" + str(getattr(ds, "PatientID", "UNKNOWN"))[:40]
    elif change == "pixel_byte":
        payload = bytearray(ds.PixelData)
        if not payload:
            raise ValueError("PixelData is empty")
        payload[len(payload) // 2] ^= 1
        ds.PixelData = bytes(payload)
    elif change == "single_pixel":
        import numpy as np
        transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        if transfer_syntax is not None and transfer_syntax.is_compressed:
            raise ValueError("single-pixel editing is not supported for compressed PixelData")
        pixels = np.asarray(ds.pixel_array).copy()
        if pixels.ndim != 2 or not np.issubdtype(pixels.dtype, np.integer):
            raise ValueError(f"Expected a single 2-D integer frame, found {pixels.shape} {pixels.dtype}")
        original_pixels = pixels.copy()
        old_value = int(pixels[0, 0])
        limits = np.iinfo(pixels.dtype)
        pixels[0, 0] = old_value + 1 if old_value < limits.max else old_value - 1
        ds.PixelData = pixels.tobytes(order="C")
    pydicom.dcmwrite(str(destination), ds, enforce_file_format=True)
    if change == "single_pixel":
        import numpy as np
        written = np.asarray(pydicom.dcmread(str(destination), force=False).pixel_array)
        changed_pixel_count = int(np.count_nonzero(written != original_pixels))
        if changed_pixel_count != 1:
            raise ValueError(f"Single-pixel edit changed {changed_pixel_count} decoded values")
    return changed_pixel_count


def _record(rows: list[dict], file: Path, scenario: str, status: str, elapsed: float,
            details: dict | None = None) -> None:
    rows.append({"file": file.name, "scenario": scenario, "status": status,
                 "latency_ms": round(elapsed * 1000, 4),
                 "details": json.dumps(details or {}, sort_keys=True)})


def render_security_report(output_dir: str | Path) -> dict:
    """Render a Markdown summary and latency plot from saved experiment CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    summary_path = output / "security_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (output / "security_scenarios.csv").open(encoding="utf-8", newline="") as stream:
        scenario_rows = list(csv.DictReader(stream))
    with (output / "security_performance.csv").open(encoding="utf-8", newline="") as stream:
        performance_rows = list(csv.DictReader(stream))
    counts = {}
    for row in scenario_rows:
        key = (row["scenario"], row["status"])
        counts[key] = counts.get(key, 0) + 1
    scenario_table = "\n".join(f"| {name} | {status} | {count} |" for (name, status), count in sorted(counts.items()))
    timing_names = [name for name in performance_rows[0] if name.endswith("_ms")]
    timing_means = {name: sum(float(row[name]) for row in performance_rows) / len(performance_rows)
                    for name in timing_names}
    timing_table = "\n".join(f"| {name.replace('_', ' ')} | {value:.4f} |" for name, value in timing_means.items())
    chart_keys = ["raw_file_sha256_ms", "metadata_sha256_ms", "pixel_data_sha256_ms",
                  "hmac_generation_ms", "hmac_verification_ms", "ed25519_signing_ms",
                  "ed25519_verification_ms", "end_to_end_verify_ms"]
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    labels = [key.replace("_ms", "").replace("_", "\n") for key in chart_keys]
    values = [timing_means[key] for key in chart_keys]
    ax.bar(labels, values, color=["#3973ac"] * 3 + ["#6b9e78"] * 2 + ["#9366a5"] * 2 + ["#d08a37"])
    ax.set_ylabel("Mean time per file (ms)")
    ax.set_title("DICOM integrity operation timings (local CPAISD test run)")
    ax.grid(axis="y", alpha=0.25)
    chart_path = output / "security_operation_timings.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    report = (
        "# DICOM Cryptographic Security Experiment\n\n"
        f"- Held-out DICOM files: {summary['input_files']}\n"
        f"- Scenario rows: {summary['scenario_rows']}\n"
        f"- Mean signed envelope: {summary['mean_manifest_bytes']:.1f} bytes\n"
        f"- Mean manifest storage overhead: {100 * summary['mean_manifest_overhead_fraction']:.4f}% of DICOM file size\n"
        "- Keys were generated temporarily for this run and deleted at exit; these results do not identify a hospital signer.\n\n"
        "## Scenario outcomes\n\n"
        "| Scenario | Outcome | Files |\n|---|---|---:|\n" + scenario_table + "\n\n"
        "An unsigned forged reference matches its attacker-modified file by construction; the signed verifier returns `MISSING_REFERENCE` because no signature is present. The single-pixel scenario asserts exactly one changed decoded value for every supported test object.\n\n"
        "## Mean operation timings\n\n"
        "| Operation | Mean time per file (ms) |\n|---|---:|\n" + timing_table + "\n\n"
        f"![Operation timing chart]({chart_path.name})\n\n"
        "These values are local prototype timings, not deployment benchmarks. The replay cache is a local stateful demo and must be protected; it is not distributed replay prevention. Pixel editing was tested only for the supported uncompressed single-frame CPAISD objects.\n"
    )
    report_path = output / "SECURITY_EXPERIMENT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    summary["timing_plot"] = str(chart_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"report": str(report_path.resolve()), "timing_plot": str(chart_path.resolve())}


def run(input_dir: str | Path, output_dir: str | Path, *, limit: int = 0) -> dict:
    paths = find_dicom_files(input_dir)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise ValueError("No readable DICOM inputs were found")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, timings, manifest_sizes, performance_rows = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="dicom_security_keys_", dir=output) as keydir:
        keydir = Path(keydir)
        private, public = keydir / "sender-private.pem", keydir / "sender-public.pem"
        registry_path, hmac_path, pseudonym_path = keydir / "trusted.json", keydir / "hmac.secret", keydir / "pseudo.secret"
        signer_id = generate_ed25519_keypair(private, public)
        add_trusted_signer(registry_path, signer_id, public)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        generate_hmac_key(hmac_path); generate_hmac_key(pseudonym_path)
        hmac_key, pseudonym_key = hmac_path.read_bytes(), pseudonym_path.read_bytes()
        private_key = load_private_key(private)
        public_key = trusted_public_key(json.loads(registry_path.read_text(encoding="utf-8")), signer_id)
        work = keydir / "scratch"; work.mkdir()

        for index, path in enumerate(paths):
            start = time.perf_counter()
            envelope = create_signed_manifest(path, private)
            signing_seconds = time.perf_counter() - start
            envelope_bytes = canonical_json_bytes(envelope)
            signed_bytes = canonical_json_bytes(envelope["manifest"])
            manifest_sizes.append(len(envelope_bytes))
            start = time.perf_counter(); valid = verify_signed_manifest(path, envelope, registry)
            verify_seconds = time.perf_counter() - start
            timings.append({"sign_ms": signing_seconds * 1000, "verify_ms": verify_seconds * 1000})
            _record(rows, path, "unchanged_signed_file", valid["status"], verify_seconds, valid["details"])

            dicom_ds = pydicom.dcmread(str(path), stop_before_pixels=False, force=False)
            operation = {"file": path.name, "manifest_bytes": len(envelope_bytes), "dicom_bytes": path.stat().st_size,
                         "manifest_overhead_fraction": len(envelope_bytes) / max(1, path.stat().st_size),
                         "manifest_create_and_sign_ms": signing_seconds * 1000,
                         "end_to_end_verify_ms": verify_seconds * 1000}
            for name, fn in (("raw_file_sha256_ms", lambda: raw_file_sha256(path)),
                             ("metadata_sha256_ms", lambda: metadata_sha256(dicom_ds)),
                             ("pixel_data_sha256_ms", lambda: pixel_data_sha256(dicom_ds))):
                start = time.perf_counter(); fn(); operation[name] = (time.perf_counter() - start) * 1000
            start = time.perf_counter(); tag = hmac_sha256(hmac_key, signed_bytes)
            operation["hmac_generation_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter(); verify_hmac_sha256(hmac_key, signed_bytes, tag)
            operation["hmac_verification_ms"] = (time.perf_counter() - start) * 1000
            import base64
            sig = base64.b64decode(envelope["signature"]["value_base64"])
            start = time.perf_counter(); private_key.sign(signed_bytes)
            operation["ed25519_signing_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter(); verify_bytes(public_key, sig, signed_bytes)
            operation["ed25519_verification_ms"] = (time.perf_counter() - start) * 1000
            performance_rows.append(operation)

            for scenario, edit in (("metadata_change", "metadata"), ("pixel_byte_flip", "pixel_byte"),
                                   ("tiny_single_pixel_change", "single_pixel")):
                edited = work / f"{index}-{scenario}.dcm"
                try:
                    changed_pixels = _write_ds(path, edited, edit)
                    start = time.perf_counter(); result = verify_signed_manifest(edited, envelope, registry)
                    details = result["details"].get("digest_mismatches", {})
                    if changed_pixels is not None:
                        details = {"changed_decoded_pixels": changed_pixels, "digest_mismatches": details}
                    _record(rows, path, scenario, result["status"], time.perf_counter() - start,
                            details)
                except Exception as exc:
                    _record(rows, path, scenario, "UNSUPPORTED", 0, {"reason": type(exc).__name__})

            unsigned_edit = work / f"{index}-forged-unsigned-reference.dcm"
            _write_ds(path, unsigned_edit, "metadata")
            forged_reference = copy.deepcopy(envelope["manifest"])
            modified_ds = pydicom.dcmread(str(unsigned_edit), stop_before_pixels=False, force=False)
            forged_reference["dicom"] = {
                "sop_instance_uid": str(getattr(modified_ds, "SOPInstanceUID", "")),
                "study_instance_uid": str(getattr(modified_ds, "StudyInstanceUID", "")),
                "series_instance_uid": str(getattr(modified_ds, "SeriesInstanceUID", "")),
            }
            forged_reference["digests"] = {
                "metadata_sha256": metadata_sha256(modified_ds),
                "pixel_data_sha256": pixel_data_sha256(modified_ds),
                "raw_file_sha256": raw_file_sha256(unsigned_edit),
            }
            naive_match = (forged_reference["digests"]["metadata_sha256"] == metadata_sha256(modified_ds)
                           and forged_reference["digests"]["pixel_data_sha256"] == pixel_data_sha256(modified_ds)
                           and forged_reference["digests"]["raw_file_sha256"] == raw_file_sha256(unsigned_edit))
            start = time.perf_counter()
            secure_result = verify_signed_manifest(unsigned_edit, {"manifest": forged_reference}, registry)
            _record(rows, path, "forged_unsigned_reference",
                    "UNSIGNED_REFERENCE_ACCEPTS" if naive_match else "UNEXPECTED_MISMATCH",
                    time.perf_counter() - start,
                    {"unauthenticated_digest_comparison_matches": naive_match,
                     "signed_verifier_status": secure_result["status"]})

            forged = copy.deepcopy(envelope)
            forged["manifest"]["digests"]["metadata_sha256"] = "0" * 64
            start = time.perf_counter(); result = verify_signed_manifest(path, forged, registry)
            _record(rows, path, "altered_unsigned_digest", result["status"], time.perf_counter() - start)
            forged_sig = copy.deepcopy(envelope)
            import base64
            signature = bytearray(base64.b64decode(forged_sig["signature"]["value_base64"])); signature[0] ^= 1
            forged_sig["signature"]["value_base64"] = base64.b64encode(signature).decode("ascii")
            start = time.perf_counter(); result = verify_signed_manifest(path, forged_sig, registry)
            _record(rows, path, "forged_signature", result["status"], time.perf_counter() - start)
            start = time.perf_counter(); result = verify_signed_manifest(path, envelope, {"schema_version": 1, "signers": {}})
            _record(rows, path, "untrusted_signer", result["status"], time.perf_counter() - start)
            unsigned = {"manifest": envelope["manifest"]}
            start = time.perf_counter(); result = verify_signed_manifest(path, unsigned, registry)
            _record(rows, path, "missing_signature", result["status"], time.perf_counter() - start)

            manifest_bytes = canonical_json_bytes(envelope["manifest"])
            tag = hmac_sha256(hmac_key, manifest_bytes)
            wrong_key_valid = verify_hmac_sha256(b"x" * 32, manifest_bytes, tag)
            _record(rows, path, "incorrect_hmac_key", "INVALID" if not wrong_key_valid else "UNEXPECTED_VALID", 0)

            reserialized = work / f"{index}-reserialized.dcm"
            try:
                serialized_ds = pydicom.dcmread(str(path), force=False)
                serialized_ds.preamble = b"R" * 128  # legal, non-pixel PS3.10 preamble change
                pydicom.dcmwrite(str(reserialized), serialized_ds, enforce_file_format=True)
                before = pydicom.dcmread(str(path), stop_before_pixels=False, force=False)
                after = pydicom.dcmread(str(reserialized), stop_before_pixels=False, force=False)
                same_metadata = metadata_sha256(before) == metadata_sha256(after)
                same_pixel = pixel_data_sha256(before) == pixel_data_sha256(after)
                raw_same = raw_file_sha256(path) == raw_file_sha256(reserialized)
                start = time.perf_counter(); result = verify_signed_manifest(reserialized, envelope, registry)
                content_only = create_signed_manifest(path, private, include_raw_file_hash=False)
                content_only_result = verify_signed_manifest(reserialized, content_only, registry, allow_content_only=True)
                _record(rows, path, "reserialization", result["status"], time.perf_counter() - start,
                        {"canonical_metadata_same": same_metadata, "pixel_bytes_same": same_pixel,
                         "raw_file_bytes_same": raw_same,
                         "raw_scoped_mismatches": result["details"].get("digest_mismatches", {}),
                         "manifest_without_raw_hash_status": content_only_result["status"]})
            except Exception as exc:
                _record(rows, path, "reserialization", "UNSUPPORTED", 0, {"reason": type(exc).__name__})

            replay_cache = work / f"replay-{index}.json"
            first = verify_signed_manifest(path, envelope, registry, replay_cache_path=replay_cache)
            start = time.perf_counter(); replay = verify_signed_manifest(path, envelope, registry, replay_cache_path=replay_cache)
            _record(rows, path, "replayed_manifest_with_local_cache", replay["status"], time.perf_counter() - start,
                    {"first_acceptance": first["status"]})

            transformed = work / f"{index}-deidentified.dcm"
            report = deidentify_research_copy(path, transformed, pseudonym_key)
            transformed_envelope = create_signed_manifest(transformed, private)
            fresh = verify_signed_manifest(transformed, transformed_envelope, registry)
            _record(rows, path, "authorized_deidentify_and_resign", fresh["status"], 0,
                    {"original_signature_valid_before_transform": valid["status"] == "VERIFIED",
                     "fresh_manifest_id": transformed_envelope["manifest"]["manifest_id"],
                     "pixel_review_required": report["pixel_review_required"]})

    csv_path = output / "security_scenarios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "scenario", "status", "latency_ms", "details"])
        writer.writeheader(); writer.writerows(rows)
    performance_path = output / "security_performance.csv"
    with performance_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(performance_rows[0]))
        writer.writeheader(); writer.writerows(performance_rows)
    counts = {}
    for row in rows:
        key = row["scenario"] + ":" + row["status"]
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "input_files": len(paths), "scenario_rows": len(rows), "scenario_counts": counts,
        "mean_signing_latency_ms": sum(item["sign_ms"] for item in timings) / len(timings),
        "mean_verification_latency_ms": sum(item["verify_ms"] for item in timings) / len(timings),
        "mean_manifest_bytes": sum(manifest_sizes) / len(manifest_sizes),
        "mean_performance_ms": {key: sum(float(item[key]) for item in performance_rows) / len(performance_rows)
                                for key in performance_rows[0] if key.endswith("_ms")},
        "mean_manifest_overhead_fraction": sum(item["manifest_overhead_fraction"] for item in performance_rows) / len(performance_rows),
        "results_csv": str(csv_path.resolve()),
        "performance_csv": str(performance_path.resolve()),
        "limitations": ["Independent temporary experiment keys; no hospital identity claims.",
                        "Metrics describe only the selected local DICOM test inputs.",
                        "The replay cache is a local stateful demo and must be protected; it is not distributed replay prevention."],
    }
    (output / "security_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render_security_report(output)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CPAISD test fold or other DICOM directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="maximum files; 0 processes every readable file")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, limit=args.limit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
