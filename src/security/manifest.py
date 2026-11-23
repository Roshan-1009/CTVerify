"""External, signed DICOM integrity manifests and verification outcomes."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pydicom

from .canonical import PROFILE, canonical_json_bytes, metadata_sha256, pixel_data_sha256
from .signing import (key_id_for_public_key, load_private_key, sign_bytes,
                      trusted_public_key, verify_bytes)

SCHEMA_VERSION = 2
DEFAULT_VALID_DAYS = 30
MAX_CLOCK_SKEW = timedelta(minutes=5)
REPLAY_LOCK_TIMEOUT_SECONDS = 10.0
STUDY_MANIFEST_DIR = "_studies"
STUDY_MANIFEST_SUFFIX = ".study-manifest.json"

OUTCOMES = {
    "VERIFIED", "INTEGRITY_MISMATCH", "INVALID_SIGNATURE", "UNTRUSTED_SIGNER",
    "MISSING_REFERENCE", "MALFORMED_DICOM", "POLICY_REVIEW_REQUIRED",
}


@dataclass(frozen=True)
class VerifiedObject:
    """A verification result and the dataset parsed from the exact bytes checked."""

    result: dict
    dataset: pydicom.Dataset | None = None
    facts: dict | None = None


def _result(status: str, *, details: dict | None = None) -> dict:
    if status not in OUTCOMES:
        raise ValueError(f"Unknown security outcome: {status}")
    return {"status": status, "details": details or {}}


def _now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be UTC")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _valid_version(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _common_fields(manifest_type: str, signer_id: str, *, recipient_id: str | None, valid_days: int,
                   object_version: int, now: datetime | None) -> dict:
    if not isinstance(valid_days, int) or isinstance(valid_days, bool) or valid_days < 1:
        raise ValueError("valid_days must be a positive integer")
    if not _valid_version(object_version):
        raise ValueError("object_version must be a positive integer")
    if recipient_id is not None and (not isinstance(recipient_id, str) or not recipient_id.strip()):
        raise ValueError("recipient_id must be a non-empty string")
    created = _now(now).replace(microsecond=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": manifest_type,
        "manifest_id": str(uuid.uuid4()),
        "created_utc": _timestamp(created),
        "expires_utc": _timestamp(created + timedelta(days=valid_days)),
        "recipient_id": recipient_id,
        "object_version": object_version,
        "canonicalization_profile": PROFILE,
        "signature_algorithm": "Ed25519",
        "signer_id": signer_id,
    }


def _envelope(content: dict, private_key) -> dict:
    signature = sign_bytes(private_key, canonical_json_bytes(content))
    return {"manifest": content,
            "signature": {"algorithm": "Ed25519", "value_base64": base64.b64encode(signature).decode("ascii")}}


def _parse_dicom(raw: bytes) -> pydicom.Dataset:
    return pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=False, force=False)


def _dicom_ids(ds: pydicom.Dataset) -> dict:
    return {
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
    }


def create_signed_manifest(dicom_path: str | Path, private_key_path: str | Path, *,
                           include_raw_file_hash: bool = True, recipient_id: str | None = None,
                           valid_days: int = DEFAULT_VALID_DAYS, object_version: int = 1,
                           now: datetime | None = None) -> dict:
    raw = Path(dicom_path).read_bytes()
    ds = _parse_dicom(raw)
    private_key = load_private_key(private_key_path)
    content = _common_fields("instance", key_id_for_public_key(private_key.public_key()),
                             recipient_id=recipient_id, valid_days=valid_days,
                             object_version=object_version, now=now)
    content["dicom"] = _dicom_ids(ds)
    content["digests"] = {
        "metadata_sha256": metadata_sha256(ds),
        "pixel_data_sha256": pixel_data_sha256(ds),
        "raw_file_sha256": hashlib.sha256(raw).hexdigest() if include_raw_file_hash else None,
    }
    return _envelope(content, private_key)


def create_signed_study_manifests(dicom_root: str | Path, private_key_path: str | Path, *,
                                  recipient_id: str | None = None, valid_days: int = DEFAULT_VALID_DAYS,
                                  object_version: int = 1, now: datetime | None = None) -> list[dict]:
    """Sign the complete instance list of every study found under ``dicom_root``."""
    from ..dicom_io import find_dicom_files

    private_key = load_private_key(private_key_path)
    signer_id = key_id_for_public_key(private_key.public_key())
    studies: dict[str, list[dict]] = {}
    for path in find_dicom_files(dicom_root):
        raw = Path(path).read_bytes()
        ds = _parse_dicom(raw)
        ids = _dicom_ids(ds)
        if not all(ids.values()):
            raise ValueError("Every DICOM object must have SOP, study and series UIDs")
        studies.setdefault(ids["study_instance_uid"], []).append({
            "sop_instance_uid": ids["sop_instance_uid"],
            "series_instance_uid": ids["series_instance_uid"],
            "metadata_sha256": metadata_sha256(ds),
            "pixel_data_sha256": pixel_data_sha256(ds),
            "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
        })
    envelopes = []
    for study_uid, instances in sorted(studies.items()):
        sop_uids = [item["sop_instance_uid"] for item in instances]
        if len(sop_uids) != len(set(sop_uids)):
            raise ValueError("A study contains duplicate SOP Instance UIDs")
        content = _common_fields("study", signer_id, recipient_id=recipient_id, valid_days=valid_days,
                                 object_version=object_version, now=now)
        content["study_instance_uid"] = study_uid
        content["instances"] = sorted(instances, key=lambda item: item["sop_instance_uid"])
        envelopes.append(_envelope(content, private_key))
    return envelopes


def save_signed_manifest(envelope: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(envelope) + b"\n")


def study_manifest_path(manifest_root: str | Path, study_uid: str) -> Path:
    token = hashlib.sha256(study_uid.encode("utf-8")).hexdigest()[:24]
    return Path(manifest_root) / STUDY_MANIFEST_DIR / f"{token}{STUDY_MANIFEST_SUFFIX}"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def load_envelope(envelope_or_path: dict | str | Path) -> Any:
    if isinstance(envelope_or_path, (str, Path)):
        return json.loads(Path(envelope_or_path).read_text(encoding="utf-8"),
                          parse_constant=_reject_json_constant)
    return envelope_or_path


def _check_envelope(envelope_or_path: dict | str | Path, trust_registry: Mapping, manifest_type: str, *,
                    expected_signer: str | None, receiver_id: str | None,
                    now: datetime | None) -> tuple[dict | None, dict | None]:
    """Authenticate an envelope, then apply its signed policy fields."""
    malformed = _result("MISSING_REFERENCE", details={"reason": "manifest or signature is missing or malformed"})
    try:
        envelope = load_envelope(envelope_or_path)
        manifest = envelope["manifest"]
        signature_info = envelope["signature"]
        if not isinstance(manifest, dict) or not isinstance(signature_info, dict):
            return None, malformed
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("canonicalization_profile") != PROFILE:
            return None, _result("POLICY_REVIEW_REQUIRED",
                                 details={"reason": "unsupported manifest schema or canonicalization profile"})
        if signature_info.get("algorithm") != "Ed25519" or manifest.get("signature_algorithm") != "Ed25519":
            return None, _result("POLICY_REVIEW_REQUIRED", details={"reason": "unsupported signature algorithm"})
        signature = base64.b64decode(signature_info["value_base64"], validate=True)
        signer_id = manifest["signer_id"]
        if not isinstance(signer_id, str):
            return None, malformed
        signed_bytes = canonical_json_bytes(manifest)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return None, malformed

    if expected_signer is not None and signer_id != expected_signer:
        return None, _result("UNTRUSTED_SIGNER", details={"reason": "manifest signer differs from expected sender",
                                                          "signer_id": signer_id})
    public_key = trusted_public_key(trust_registry, signer_id)
    if public_key is None:
        return None, _result("UNTRUSTED_SIGNER", details={"reason": "signer is not in the trusted-key registry",
                                                          "signer_id": signer_id})
    if not verify_bytes(public_key, signature, signed_bytes):
        return None, _result("INVALID_SIGNATURE", details={"reason": "manifest signature does not verify"})

    def policy(reason: str) -> tuple[None, dict]:
        return None, _result("POLICY_REVIEW_REQUIRED", details={"reason": reason})

    if manifest.get("manifest_type") != manifest_type:
        return policy(f"manifest is not a signed {manifest_type} manifest")
    recipient = manifest.get("recipient_id")
    if recipient is None and receiver_id is not None:
        return policy("manifest is not bound to this receiver")
    if recipient is not None and recipient != receiver_id:
        return policy("manifest is bound to a different receiver")
    if not _valid_version(manifest.get("object_version")):
        return policy("manifest has an invalid object version")
    try:
        created, expires = _parse_timestamp(manifest.get("created_utc")), _parse_timestamp(manifest.get("expires_utc"))
        current = _now(now)
    except (ValueError, TypeError):
        return policy("manifest has an invalid validity period")
    if created > current + MAX_CLOCK_SKEW:
        return policy("manifest creation time is in the future")
    if current > expires or expires <= created:
        return policy("manifest has expired")
    return manifest, None


@contextmanager
def _replay_lock(cache_path: Path):
    """Hold an OS-level lock, which is released automatically if the process exits."""
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    deadline = time.monotonic() + REPLAY_LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the replay cache lock")
                time.sleep(0.01)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _read_replay_state(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {"schema_version": SCHEMA_VERSION, "accepted_manifest_ids": [], "latest_object_versions": {}}
    state = json.loads(cache_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(state, dict):
        raise ValueError("Replay cache must be a JSON object")
    state.setdefault("accepted_manifest_ids", [])
    state.setdefault("latest_object_versions", {})
    if not isinstance(state["accepted_manifest_ids"], list) or not isinstance(state["latest_object_versions"], dict):
        raise ValueError("Replay cache has an invalid structure")
    return state


def _write_replay_state(cache_path: Path, state: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=cache_path.name + ".", suffix=".tmp", dir=cache_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(state, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def replay_transaction(cache_path: str | Path, entries: Iterable[dict], *, commit: bool = True) -> dict[str, str]:
    """Atomically check entries and, if all pass and ``commit`` is true, record them.

    Returns a mapping of conflicting manifest IDs to reasons. Nothing is written
    when any entry conflicts.
    """
    path = Path(cache_path)
    entries = list(entries)
    with _replay_lock(path):
        state = _read_replay_state(path)
        accepted = set(state["accepted_manifest_ids"])
        latest = dict(state["latest_object_versions"])
        conflicts: dict[str, str] = {}
        batch: list[str] = []
        for entry in entries:
            manifest_id, key, version = entry["manifest_id"], entry["object_key"], entry["object_version"]
            if manifest_id in accepted or manifest_id in batch:
                conflicts[manifest_id] = "replayed manifest identifier"
            elif version <= latest.get(key, 0):
                conflicts[manifest_id] = "object version is not newer than the accepted version"
            else:
                batch.append(manifest_id)
                latest[key] = version
        if conflicts or not commit:
            return conflicts
        state["schema_version"] = SCHEMA_VERSION
        state["accepted_manifest_ids"] = state["accepted_manifest_ids"] + batch
        state["latest_object_versions"] = latest
        _write_replay_state(path, state)
    return {}


def verify_instance(dicom_path: str | Path, envelope_or_path: dict | str | Path, trust_registry: Mapping, *,
                    expected_signer: str | None = None, receiver_id: str | None = None,
                    replay_cache_path: str | Path | None = None, allow_content_only: bool = False,
                    now: datetime | None = None) -> VerifiedObject:
    """Read a file once, verify those bytes, and return the dataset parsed from them."""
    raw, ds, read_error = None, None, None
    try:
        raw = Path(dicom_path).read_bytes()
        ds = _parse_dicom(raw)
    except Exception as exc:
        read_error = type(exc).__name__

    manifest, failure = _check_envelope(envelope_or_path, trust_registry, "instance",
                                        expected_signer=expected_signer, receiver_id=receiver_id, now=now)
    if failure is not None:
        return VerifiedObject(failure, ds)
    if read_error is not None:
        return VerifiedObject(_result("MALFORMED_DICOM", details={"reason": read_error}))

    expected_ids, expected_digests = manifest.get("dicom"), manifest.get("digests")
    if not isinstance(expected_ids, dict) or not isinstance(expected_digests, dict):
        return VerifiedObject(_result("POLICY_REVIEW_REQUIRED",
                                      details={"reason": "required manifest sections are absent"}), ds)
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id or not all(
            expected_ids.get(key) for key in ("sop_instance_uid", "study_instance_uid", "series_instance_uid")):
        return VerifiedObject(_result("MISSING_REFERENCE", details={"reason": "manifest lacks required identifiers"}), ds)
    if not all(isinstance(expected_digests.get(key), str) and len(expected_digests[key]) == 64
               for key in ("metadata_sha256", "pixel_data_sha256")):
        return VerifiedObject(_result("MISSING_REFERENCE",
                                      details={"reason": "manifest lacks required content digests"}), ds)
    raw_expected = expected_digests.get("raw_file_sha256")
    if raw_expected is None and not allow_content_only:
        return VerifiedObject(_result("POLICY_REVIEW_REQUIRED", details={
            "reason": "manifest does not authenticate the raw file, including its preamble"}), ds)

    actual_ids = _dicom_ids(ds)
    try:
        actual = {"metadata_sha256": metadata_sha256(ds), "pixel_data_sha256": pixel_data_sha256(ds)}
    except (ValueError, TypeError) as exc:
        return VerifiedObject(_result("MALFORMED_DICOM", details={"reason": str(exc)}), ds)
    raw_digest = hashlib.sha256(raw).hexdigest()
    if raw_expected is not None:
        actual["raw_file_sha256"] = raw_digest
    mismatches = {scope: {"expected": expected_digests.get(scope), "actual": value}
                  for scope, value in actual.items() if expected_digests.get(scope) != value}
    id_mismatches = {key: {"expected": expected_ids.get(key), "actual": value}
                     for key, value in actual_ids.items() if expected_ids.get(key) != value}
    if mismatches or id_mismatches:
        return VerifiedObject(_result("INTEGRITY_MISMATCH", details={"digest_mismatches": mismatches,
                                                                     "identifier_mismatches": id_mismatches}), ds)

    entry = {"manifest_id": manifest_id, "object_key": f"sop:{actual_ids['sop_instance_uid']}",
             "object_version": manifest["object_version"]}
    if replay_cache_path is not None:
        try:
            conflicts = replay_transaction(replay_cache_path, [entry])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return VerifiedObject(_result("POLICY_REVIEW_REQUIRED",
                                          details={"reason": "replay cache is unreadable or unavailable"}), ds)
        if conflicts:
            return VerifiedObject(_result("POLICY_REVIEW_REQUIRED", details={
                "reason": conflicts[manifest_id], "manifest_id": manifest_id}), ds)

    result = _result("VERIFIED", details={
        "signer_id": manifest["signer_id"], "verified_scopes": list(actual), "signature_valid": True,
        "object_version": manifest["object_version"], "expires_utc": manifest["expires_utc"],
        "preamble_authenticated": raw_expected is not None,
        "protections": {"replay_cache": replay_cache_path is not None,
                        "expected_signer": expected_signer is not None,
                        "recipient_binding": receiver_id is not None},
    })
    facts = {**actual_ids, **actual, "raw_file_sha256": raw_digest, "signer_id": manifest["signer_id"],
             "replay_entry": entry}
    return VerifiedObject(result, ds, facts)


def verify_signed_manifest(dicom_path: str | Path, envelope_or_path: dict | str | Path,
                           trust_registry: Mapping, *, expected_signer: str | None = None,
                           receiver_id: str | None = None, replay_cache_path: str | Path | None = None,
                           allow_content_only: bool = False, now: datetime | None = None) -> dict:
    """Verify trust/signature first, then independently compare each digest scope."""
    return verify_instance(dicom_path, envelope_or_path, trust_registry, expected_signer=expected_signer,
                           receiver_id=receiver_id, replay_cache_path=replay_cache_path,
                           allow_content_only=allow_content_only, now=now).result


def verify_study_envelope(envelope_or_path: dict | str | Path, trust_registry: Mapping, *,
                          expected_signer: str | None = None, receiver_id: str | None = None,
                          now: datetime | None = None) -> tuple[dict | None, dict | None]:
    """Return an authenticated study manifest, or a failure result."""
    manifest, failure = _check_envelope(envelope_or_path, trust_registry, "study",
                                        expected_signer=expected_signer, receiver_id=receiver_id, now=now)
    if failure is not None:
        return None, failure
    if not isinstance(manifest.get("manifest_id"), str) or not manifest["manifest_id"] \
            or not isinstance(manifest.get("study_instance_uid"), str) or not manifest["study_instance_uid"] \
            or not isinstance(manifest.get("instances"), list) or not manifest["instances"]:
        return None, _result("MISSING_REFERENCE", details={"reason": "study manifest lacks required fields"})
    return manifest, None


def compare_study_instances(manifest: dict, facts: list[dict]) -> dict:
    """Compare verified instances with the complete list signed for their study."""
    fields = ("series_instance_uid", "metadata_sha256", "pixel_data_sha256", "raw_file_sha256")
    expected: dict[str, dict] = {}
    for item in manifest["instances"]:
        sop = item.get("sop_instance_uid") if isinstance(item, dict) else None
        if not isinstance(sop, str) or not sop or sop in expected:
            return _result("POLICY_REVIEW_REQUIRED", details={"reason": "study manifest has an invalid instance list"})
        expected[sop] = item
    observed: dict[str, dict] = {}
    duplicates = 0
    for fact in facts:
        if fact["sop_instance_uid"] in observed:
            duplicates += 1
        observed[fact["sop_instance_uid"]] = fact
    missing = set(expected) - set(observed)
    unexpected = set(observed) - set(expected)
    changed = sum(1 for sop in set(expected) & set(observed)
                  if any(expected[sop].get(field) != observed[sop].get(field) for field in fields))
    wrong_signer = sum(1 for fact in facts if fact["signer_id"] != manifest["signer_id"])
    counts = {"expected_instances": len(expected), "verified_instances": len(observed),
              "missing_instances": len(missing), "unexpected_instances": len(unexpected),
              "changed_instances": changed, "duplicate_instances": duplicates,
              "instances_from_other_signer": wrong_signer, "signer_id": manifest["signer_id"]}
    complete = not (missing or unexpected or changed or duplicates or wrong_signer)
    return _result("VERIFIED" if complete else "INTEGRITY_MISMATCH", details=counts)
