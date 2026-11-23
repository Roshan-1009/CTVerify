"""Command line interface: ``python -m src.security.cli --help``."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .audit import append_audit_event
from .canonical import canonical_json_bytes, raw_file_sha256
from .manifest import (DEFAULT_VALID_DAYS, create_signed_manifest, create_signed_study_manifests,
                       load_envelope, save_signed_manifest, study_manifest_path, verify_signed_manifest)
from .privacy import (decrypt_file, deidentify_research_copy, encrypt_file,
                      generate_aes256_key)
from .signing import (add_trusted_signer, generate_ed25519_keypair,
                      generate_hmac_key, hmac_sha256, revoke_trusted_signer,
                      verify_hmac_sha256)
from .workflow import verify_directory_then_analyze, verify_then_analyze


def _key_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _add_signing_policy(parser) -> None:
    parser.add_argument("--recipient-id", help="bind the signed manifest to this receiver")
    parser.add_argument("--valid-days", type=int, default=DEFAULT_VALID_DAYS, help="days until the manifest expires")
    parser.add_argument("--object-version", type=int, default=1, help="increase when reissuing a corrected object")


def _add_verification_policy(parser) -> None:
    parser.add_argument("--expected-signer", help="accept only this signer ID")
    parser.add_argument("--receiver-id", help="this receiver's ID, required for recipient-bound manifests")
    parser.add_argument("--replay-cache", help="stateful manifest-ID and object-version cache; protect it against tampering")


def _verification_policy(args) -> dict:
    return {"expected_signer": args.expected_signer, "receiver_id": args.receiver_id,
            "replay_cache_path": args.replay_cache}


def _signer_of(envelope) -> str | None:
    manifest = envelope.get("manifest") if isinstance(envelope, dict) else None
    signer = manifest.get("signer_id") if isinstance(manifest, dict) else None
    return signer if isinstance(signer, str) else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DICOM signed-manifest and AI-assisted integrity prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("keygen", help="create a local Ed25519 signer keypair")
    p.add_argument("--private-key", required=True); p.add_argument("--public-key", required=True)
    p = sub.add_parser("trust-add", help="explicitly add a public key to a trusted registry")
    p.add_argument("--registry", required=True); p.add_argument("--signer-id", required=True); p.add_argument("--public-key", required=True)
    p = sub.add_parser("trust-revoke", help="deactivate a signer in the local trusted registry")
    p.add_argument("--registry", required=True); p.add_argument("--signer-id", required=True)
    p = sub.add_parser("hmac-keygen", help="generate a 256-bit shared HMAC key")
    p.add_argument("--key", required=True)
    p = sub.add_parser("pseudonym-keygen", help="generate a separate 256-bit pseudonymization key")
    p.add_argument("--key", required=True)
    p = sub.add_parser("aes-keygen", help="generate an AES-256-GCM file encryption key")
    p.add_argument("--key", required=True)
    p = sub.add_parser("sign", help="sign hashes for one DICOM file")
    p.add_argument("--dicom", required=True); p.add_argument("--private-key", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--hmac-key", help="optional shared key for an independent HMAC experiment")
    _add_signing_policy(p)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-signer")
    p = sub.add_parser("sign-directory", help="sign every DICOM file and a complete manifest for each study")
    p.add_argument("--dicom-dir", required=True); p.add_argument("--private-key", required=True); p.add_argument("--manifest-dir", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-signer")
    _add_signing_policy(p)
    p = sub.add_parser("verify", help="verify trust, manifest signature, and DICOM hashes")
    p.add_argument("--dicom", required=True); p.add_argument("--manifest", required=True); p.add_argument("--trust-registry", required=True)
    p.add_argument("--hmac-key"); p.add_argument("--audit-log", default="reports/security_audit.jsonl")
    _add_verification_policy(p)
    p.add_argument("--actor-id", default="local-verifier")
    p = sub.add_parser("analyze", help="verify first, then run existing metadata/rule detectors")
    p.add_argument("--dicom", required=True); p.add_argument("--manifest", required=True); p.add_argument("--trust-registry", required=True)
    p.add_argument("--if-model"); p.add_argument("--ae-model"); p.add_argument("--pixel-model"); p.add_argument("--unet-checkpoint")
    p.add_argument("--allow-unverified-ml", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-verifier")
    _add_verification_policy(p)
    p = sub.add_parser("analyze-directory", help="verify per-file manifests then run study/series-aware analysis")
    p.add_argument("--dicom-dir", required=True); p.add_argument("--manifest-dir", required=True)
    p.add_argument("--trust-registry", required=True); p.add_argument("--if-model"); p.add_argument("--ae-model")
    p.add_argument("--pixel-model"); p.add_argument("--unet-checkpoint"); p.add_argument("--output", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-verifier")
    _add_verification_policy(p)
    p = sub.add_parser("deidentify", help="write a pseudonymized research copy (limited transformation)")
    p.add_argument("--dicom", required=True); p.add_argument("--output", required=True); p.add_argument("--pseudonym-key", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-transform")
    p = sub.add_parser("encrypt", help="AES-GCM encrypt a file at rest")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--key", required=True)
    p = sub.add_parser("decrypt", help="AES-GCM decrypt a protected file")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--key", required=True)
    p = sub.add_parser("transform", help="verify, de-identify a copy, and sign the transformed object")
    p.add_argument("--dicom", required=True); p.add_argument("--manifest", required=True); p.add_argument("--trust-registry", required=True)
    p.add_argument("--output", required=True); p.add_argument("--output-manifest", required=True)
    p.add_argument("--pseudonym-key", required=True); p.add_argument("--private-key", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="authorized-transformer")
    _add_verification_policy(p)
    p.add_argument("--output-recipient-id", help="bind the transformed object's manifest to this receiver")
    p.add_argument("--output-valid-days", type=int, default=DEFAULT_VALID_DAYS)
    p.add_argument("--output-object-version", type=int, default=1)
    args = parser.parse_args(argv)

    if args.command == "keygen":
        print(json.dumps({"signer_id": generate_ed25519_keypair(args.private_key, args.public_key),
                          "private_key_path": str(Path(args.private_key).resolve()), "public_key_path": str(Path(args.public_key).resolve())}, indent=2))
    elif args.command == "trust-add":
        add_trusted_signer(args.registry, args.signer_id, args.public_key); print(f"Trusted signer registered: {args.signer_id}")
    elif args.command == "trust-revoke":
        revoke_trusted_signer(args.registry, args.signer_id); print(f"Signer deactivated: {args.signer_id}")
    elif args.command == "hmac-keygen":
        generate_hmac_key(args.key); print(f"Generated HMAC key: {Path(args.key).resolve()}")
    elif args.command == "pseudonym-keygen":
        generate_hmac_key(args.key); print(f"Generated pseudonymization key: {Path(args.key).resolve()}")
    elif args.command == "aes-keygen":
        generate_aes256_key(args.key); print(f"Generated AES-256 key: {Path(args.key).resolve()}")
    elif args.command == "sign":
        envelope = create_signed_manifest(args.dicom, args.private_key, recipient_id=args.recipient_id,
                                          valid_days=args.valid_days, object_version=args.object_version)
        save_signed_manifest(envelope, args.manifest)
        if args.hmac_key:
            tag = hmac_sha256(_key_bytes(args.hmac_key), canonical_json_bytes(envelope["manifest"]))
            Path(str(args.manifest) + ".hmac").write_text(json.dumps({"algorithm": "HMAC-SHA256", "tag": tag}) + "\n", encoding="utf-8")
        append_audit_event(args.audit_log, "sign_manifest", "SIGNED", actor_id=args.actor_id,
                           source_sha256=raw_file_sha256(args.dicom), output_sha256=raw_file_sha256(args.manifest),
                           details={"signer_id": envelope["manifest"]["signer_id"], "hmac_created": bool(args.hmac_key)})
        print(json.dumps({"manifest": args.manifest, "signer_id": envelope["manifest"]["signer_id"],
                          "hmac_sidecar": str(args.manifest) + ".hmac" if args.hmac_key else None}, indent=2))
    elif args.command == "sign-directory":
        from ..dicom_io import find_dicom_files, relative_key
        policy = {"recipient_id": args.recipient_id, "valid_days": args.valid_days, "object_version": args.object_version}
        files = find_dicom_files(args.dicom_dir)
        for path in files:
            sidecar = Path(args.manifest_dir) / (relative_key(path, args.dicom_dir) + ".manifest.json")
            save_signed_manifest(create_signed_manifest(path, args.private_key, **policy), sidecar)
        studies = create_signed_study_manifests(args.dicom_dir, args.private_key, **policy)
        for envelope in studies:
            save_signed_manifest(envelope, study_manifest_path(args.manifest_dir, envelope["manifest"]["study_instance_uid"]))
        append_audit_event(args.audit_log, "sign_directory", "SIGNED", actor_id=args.actor_id,
                           details={"files_signed": len(files), "studies_signed": len(studies)})
        print(json.dumps({"files_signed": len(files), "studies_signed": len(studies),
                          "manifest_dir": str(Path(args.manifest_dir).resolve())}, indent=2))
    elif args.command == "verify":
        try:
            envelope = load_envelope(args.manifest)
        except (OSError, ValueError):
            envelope = None
        registry = json.loads(Path(args.trust_registry).read_text(encoding="utf-8"))
        hmac_status = None
        if args.hmac_key:
            hmac_sidecar = Path(str(args.manifest) + ".hmac")
            try:
                sidecar = json.loads(hmac_sidecar.read_text(encoding="utf-8"))
                valid = verify_hmac_sha256(_key_bytes(args.hmac_key), canonical_json_bytes(envelope["manifest"]), sidecar["tag"])
                hmac_status = "VALID" if valid else "INVALID"
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                hmac_status = "MISSING_OR_INVALID"
        policy = _verification_policy(args)
        if hmac_status not in (None, "VALID"):
            policy["replay_cache_path"] = None  # A rejected delivery must not consume the manifest ID.
        result = verify_signed_manifest(args.dicom, args.manifest, registry, **policy)
        if hmac_status is not None:
            result["hmac"] = {"status": hmac_status}
        final_status = "HMAC_FAILED" if result["status"] == "VERIFIED" and hmac_status not in (None, "VALID") else result["status"]
        result["final_status"] = final_status
        source_digest = raw_file_sha256(args.dicom) if Path(args.dicom).is_file() else None
        append_audit_event(args.audit_log, "verify_manifest", final_status, actor_id=args.actor_id,
                           source_sha256=source_digest,
                           details={"signer_id": _signer_of(envelope), "signature_status": result["status"],
                                    "hmac_status": hmac_status})
        print(json.dumps(result, indent=2))
        return 0 if final_status == "VERIFIED" else 2
    elif args.command == "analyze":
        result = verify_then_analyze(args.dicom, args.manifest, args.trust_registry,
                                     if_model_path=args.if_model, autoencoder_model_path=args.ae_model,
                                     pixel_model_path=args.pixel_model, unet_checkpoint_path=args.unet_checkpoint,
                                     run_ml_on_unverified=args.allow_unverified_ml, **_verification_policy(args))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(result, indent=2) + "\n"
        Path(args.output).write_text(serialized, encoding="utf-8")
        crypto_status = result["cryptographic"]["status"]
        append_audit_event(args.audit_log, "verify_and_analyze", crypto_status, actor_id=args.actor_id,
                           source_sha256=raw_file_sha256(args.dicom) if Path(args.dicom).is_file() else None,
                           details={"signer_id": result["cryptographic"]["details"].get("signer_id"),
                                    "ml_status": result.get("ml_status"),
                                    "report_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()})
        print(json.dumps(result, indent=2))
        return 0 if crypto_status == "VERIFIED" else 2
    elif args.command == "analyze-directory":
        result = verify_directory_then_analyze(args.dicom_dir, args.manifest_dir, args.trust_registry,
                                               if_model_path=args.if_model, autoencoder_model_path=args.ae_model,
                                               pixel_model_path=args.pixel_model,
                                               unet_checkpoint_path=args.unet_checkpoint, **_verification_policy(args))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(result, indent=2) + "\n"
        Path(args.output).write_text(serialized, encoding="utf-8")
        status_counts = {}
        for item in result["results"].values():
            status = item["cryptographic"]["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        study_counts = {}
        for item in result["study_verification"].values():
            study_counts[item["status"]] = study_counts.get(item["status"], 0) + 1
        append_audit_event(args.audit_log, "verify_directory_and_analyze", result["overall_status"], actor_id=args.actor_id,
                           details={"files_discovered": result["files_discovered"],
                                    "files_verified": result["files_verified"],
                                    "study_status_counts": study_counts,
                                    "crypto_status_counts": status_counts,
                                    "report_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()})
        print(json.dumps({"overall_status": result["overall_status"], "files_discovered": result["files_discovered"],
                          "files_verified": result["files_verified"],
                          "files_not_verified": result.get("files_not_verified", 0),
                          "report": str(Path(args.output).resolve()), "crypto_status_counts": status_counts}, indent=2))
        return 0 if result["overall_status"] == "VERIFIED" else 2
    elif args.command == "deidentify":
        report = deidentify_research_copy(args.dicom, args.output, _key_bytes(args.pseudonym_key))
        append_audit_event(args.audit_log, "deidentify_research_copy", "TRANSFORMED", actor_id=args.actor_id,
                           source_sha256=report["source_sha256"], output_sha256=report["output_sha256"],
                           details={"pixel_review_required": report["pixel_review_required"]})
        print(json.dumps(report, indent=2))
    elif args.command == "encrypt":
        print(json.dumps(encrypt_file(args.input, args.output, _key_bytes(args.key)), indent=2))
    elif args.command == "decrypt":
        decrypt_file(args.input, args.output, _key_bytes(args.key)); print(f"Decrypted file: {args.output}")
    elif args.command == "transform":
        registry = json.loads(Path(args.trust_registry).read_text(encoding="utf-8"))
        verification = verify_signed_manifest(args.dicom, args.manifest, registry, **_verification_policy(args))
        if verification["status"] != "VERIFIED":
            print(json.dumps({"original_verification": verification, "transformation": "NOT_PERFORMED"}, indent=2))
            return 2
        report = deidentify_research_copy(args.dicom, args.output, _key_bytes(args.pseudonym_key))
        envelope = create_signed_manifest(args.output, args.private_key, recipient_id=args.output_recipient_id,
                                          valid_days=args.output_valid_days, object_version=args.output_object_version)
        save_signed_manifest(envelope, args.output_manifest)
        append_audit_event(args.audit_log, "verify_deidentify_resign", "TRANSFORMED_AND_RESIGNED", actor_id=args.actor_id,
                           source_sha256=report["source_sha256"], output_sha256=report["output_sha256"],
                           details={"original_manifest_id": json.loads(Path(args.manifest).read_text(encoding="utf-8")).get("manifest", {}).get("manifest_id"),
                                    "new_manifest_id": envelope["manifest"]["manifest_id"],
                                    "new_signer_id": envelope["manifest"]["signer_id"],
                                    "pixel_review_required": report["pixel_review_required"]})
        print(json.dumps({"original_verification": verification, "transformation": report,
                          "new_manifest": args.output_manifest, "new_signer_id": envelope["manifest"]["signer_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
