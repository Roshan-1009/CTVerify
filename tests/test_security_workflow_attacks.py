"""Regression tests for attacks found by the Tamarin protocol model."""
import contextlib
import io
import json
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from pydicom import dcmread, dcmwrite
from pydicom.uid import generate_uid

from scripts.make_demo_dataset import create_dataset
from src.security import cli, manifest as mf, workflow as wf
from src.security.manifest import (create_signed_manifest, create_signed_study_manifests,
                                   save_signed_manifest, study_manifest_path, verify_signed_manifest)
from src.security.signing import add_trusted_signer, generate_ed25519_keypair, generate_hmac_key


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        value = function(*args, **kwargs)
    return value, stream.getvalue()


class WorkflowAttackRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dicom = self.root / "original.dcm"
        create_dataset(self.dicom, generate_uid(), generate_uid(), 1, "PHANTOM001")
        self.a_private, a_public = self.root / "a.pem", self.root / "a.pub"
        self.b_private, b_public = self.root / "b.pem", self.root / "b.pub"
        self.a_id = generate_ed25519_keypair(self.a_private, a_public)
        self.b_id = generate_ed25519_keypair(self.b_private, b_public)
        self.registry = self.root / "registry.json"
        add_trusted_signer(self.registry, self.a_id, a_public)
        add_trusted_signer(self.registry, self.b_id, b_public)
        self.trust = json.loads(self.registry.read_text(encoding="utf-8"))
        self.envelope = create_signed_manifest(self.dicom, self.a_private)
        self.manifest = self.root / "original.manifest.json"
        save_signed_manifest(self.envelope, self.manifest)
        self.tampered = self.rewrite("tampered.dcm", lambda ds: setattr(ds, "Modality", "MR"))

    def tearDown(self):
        self.tmp.cleanup()

    def rewrite(self, name, change):
        target = self.root / name
        ds = dcmread(self.dicom)
        change(ds)
        dcmwrite(target, ds, enforce_file_format=True)
        return target

    def make_study(self, count=3):
        dicom_dir, manifest_dir = self.root / "study", self.root / "sidecars"
        study, series = generate_uid(), generate_uid()
        for number in range(1, count + 1):
            path = dicom_dir / f"slice-{number}.dcm"
            create_dataset(path, study, series, number, "PHANTOM-S")
            save_signed_manifest(create_signed_manifest(path, self.a_private),
                                 manifest_dir / f"{path.name}.manifest.json")
        for envelope in create_signed_study_manifests(dicom_dir, self.a_private):
            save_signed_manifest(envelope, study_manifest_path(manifest_dir, envelope["manifest"]["study_instance_uid"]))
        return dicom_dir, manifest_dir

    def test_analysis_path_rejects_replay(self):
        cache = self.root / "cache.json"
        first = wf.verify_then_analyze(self.dicom, self.manifest, self.registry, replay_cache_path=cache)
        second = wf.verify_then_analyze(self.dicom, self.manifest, self.registry, replay_cache_path=cache)
        self.assertEqual(first["cryptographic"]["status"], "VERIFIED")
        self.assertEqual(second["cryptographic"]["status"], "POLICY_REVIEW_REQUIRED")
        self.assertEqual(second["ml_status"], "SKIPPED_CRYPTOGRAPHIC_POLICY")

    def test_strict_cli_requires_all_receiver_settings(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as result:
            cli.main(["verify", "--strict", "--dicom", str(self.dicom), "--manifest",
                      str(self.manifest), "--trust-registry", str(self.registry)])
        self.assertEqual(result.exception.code, 2)
        self.assertIn("--strict requires --expected-signer, --receiver-id, --replay-cache",
                      errors.getvalue())

    def test_strict_cli_accepts_configured_delivery(self):
        save_signed_manifest(create_signed_manifest(self.dicom, self.a_private, recipient_id="receiver"),
                             self.manifest)
        status, output = quiet(cli.main, ["verify", "--strict", "--dicom", str(self.dicom),
                                          "--manifest", str(self.manifest), "--trust-registry",
                                          str(self.registry), "--expected-signer", self.a_id,
                                          "--receiver-id", "receiver", "--replay-cache",
                                          str(self.root / "strict-cache.json"),
                                          "--audit-log", str(self.root / "audit.jsonl")])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output)["final_status"], "VERIFIED")
        self.assertTrue((self.root / "strict-cache.json").is_file())

    def test_strict_analysis_rejects_unverified_ml_override(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as result:
            cli.main(["analyze", "--strict", "--allow-unverified-ml", "--dicom", str(self.dicom),
                      "--manifest", str(self.manifest), "--trust-registry", str(self.registry),
                      "--expected-signer", self.a_id, "--receiver-id", "receiver",
                      "--replay-cache", str(self.root / "strict-cache.json")])
        self.assertEqual(result.exception.code, 2)
        self.assertIn("--strict cannot be combined with --allow-unverified-ml", errors.getvalue())

    def test_overlapping_cache_checks_accept_once(self):
        cache, results = self.root / "race.json", []
        barrier, real_parse = threading.Barrier(2, timeout=20), mf._parse_dicom

        def held_parse(raw):
            parsed = real_parse(raw)
            barrier.wait()
            return parsed

        def worker():
            results.append(verify_signed_manifest(self.dicom, self.envelope, self.trust,
                                                  replay_cache_path=cache)["status"])

        with mock.patch.object(mf, "_parse_dicom", held_parse):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        self.assertEqual(sorted(results), ["POLICY_REVIEW_REQUIRED", "VERIFIED"])

    def test_corrupt_replay_cache_fails_closed(self):
        cache = self.root / "corrupt.json"
        cache.write_text("[]", encoding="utf-8")
        result = verify_signed_manifest(self.dicom, self.envelope, self.trust, replay_cache_path=cache)
        self.assertEqual(result["status"], "POLICY_REVIEW_REQUIRED")

    def test_manifest_is_bound_to_recipient(self):
        bound = create_signed_manifest(self.dicom, self.a_private, recipient_id="hospital-b")
        check = lambda envelope, receiver: verify_signed_manifest(self.dicom, envelope, self.trust,
                                                                  receiver_id=receiver)["status"]
        self.assertEqual(check(bound, "hospital-b"), "VERIFIED")
        self.assertEqual(check(bound, "hospital-c"), "POLICY_REVIEW_REQUIRED")
        self.assertEqual(check(bound, None), "POLICY_REVIEW_REQUIRED")
        self.assertEqual(check(self.envelope, "hospital-b"), "POLICY_REVIEW_REQUIRED")

    def test_superseded_version_is_rejected(self):
        corrected = self.rewrite("corrected.dcm", lambda ds: setattr(ds, "SliceThickness", 2.0))
        new_envelope = create_signed_manifest(corrected, self.a_private, object_version=2)
        cache = self.root / "versions.json"
        self.assertEqual(verify_signed_manifest(corrected, new_envelope, self.trust,
                                                replay_cache_path=cache)["status"], "VERIFIED")
        older = verify_signed_manifest(self.dicom, self.envelope, self.trust, replay_cache_path=cache)
        self.assertEqual(older["status"], "POLICY_REVIEW_REQUIRED")
        self.assertIn("not newer", older["details"]["reason"])

    def test_expired_and_future_manifests_are_rejected(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        envelope = create_signed_manifest(self.dicom, self.a_private, valid_days=1, now=start)
        check = lambda when: verify_signed_manifest(self.dicom, envelope, self.trust, now=when)["status"]
        self.assertEqual(check(start + timedelta(hours=23)), "VERIFIED")
        self.assertEqual(check(start + timedelta(days=1, seconds=1)), "POLICY_REVIEW_REQUIRED")
        self.assertEqual(check(start - timedelta(hours=1)), "POLICY_REVIEW_REQUIRED")

    def test_old_schema_and_type_confusion_are_rejected(self):
        legacy = json.loads(json.dumps(self.envelope))
        legacy["manifest"]["schema_version"] = 1
        self.assertEqual(verify_signed_manifest(self.dicom, legacy, self.trust)["status"], "POLICY_REVIEW_REQUIRED")
        dicom_dir, _ = self.make_study(1)
        study = create_signed_study_manifests(dicom_dir, self.a_private)[0]
        self.assertEqual(verify_signed_manifest(dicom_dir / "slice-1.dcm", study, self.trust)["status"],
                         "POLICY_REVIEW_REQUIRED")

    def test_complete_study_verifies(self):
        dicom_dir, manifest_dir = self.make_study()
        result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
        self.assertEqual(result["overall_status"], "VERIFIED")
        self.assertEqual(result["files_verified"], 3)

    def test_deleted_slice_fails_study_verification(self):
        dicom_dir, manifest_dir = self.make_study()
        (dicom_dir / "slice-2.dcm").unlink()
        result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
        study = next(iter(result["study_verification"].values()))
        self.assertEqual(result["overall_status"], "NOT_VERIFIED")
        self.assertEqual(study["status"], "INTEGRITY_MISMATCH")
        self.assertEqual(study["details"]["missing_instances"], 1)

    def test_deleted_study_manifest_cannot_pass(self):
        dicom_dir, manifest_dir = self.make_study()
        shutil.rmtree(manifest_dir / mf.STUDY_MANIFEST_DIR)
        result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
        self.assertEqual(result["overall_status"], "NOT_VERIFIED")
        self.assertEqual(next(iter(result["study_verification"].values()))["status"], "MISSING_REFERENCE")

    def test_entirely_deleted_study_is_reported(self):
        dicom_dir, manifest_dir = self.make_study()
        other = self.root / "other.dcm"
        create_dataset(other, generate_uid(), generate_uid(), 1, "PHANTOM-O")
        for path in dicom_dir.glob("*.dcm"):
            path.unlink()
        shutil.copy2(other, dicom_dir / "other.dcm")
        save_signed_manifest(create_signed_manifest(dicom_dir / "other.dcm", self.a_private),
                             manifest_dir / "other.dcm.manifest.json")
        for envelope in create_signed_study_manifests(dicom_dir, self.a_private):
            save_signed_manifest(envelope, study_manifest_path(manifest_dir, envelope["manifest"]["study_instance_uid"]))
        result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
        self.assertEqual(result["overall_status"], "NOT_VERIFIED")
        self.assertTrue(any(item["details"].get("missing_instances") == 3
                            for item in result["study_verification"].values()))

    def test_added_slice_fails_study_verification(self):
        dicom_dir, manifest_dir = self.make_study()
        first = dcmread(dicom_dir / "slice-1.dcm")
        extra = dicom_dir / "slice-extra.dcm"
        create_dataset(extra, str(first.StudyInstanceUID), str(first.SeriesInstanceUID), 9, "PHANTOM-S")
        save_signed_manifest(create_signed_manifest(extra, self.a_private),
                             manifest_dir / "slice-extra.dcm.manifest.json")
        result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
        self.assertEqual(result["overall_status"], "NOT_VERIFIED")
        self.assertEqual(next(iter(result["study_verification"].values()))["details"]["unexpected_instances"], 1)

    def test_failed_directory_does_not_consume_replay_state(self):
        dicom_dir, manifest_dir = self.make_study()
        cache = self.root / "directory-cache.json"
        held = self.root / "held.dcm"
        shutil.move(dicom_dir / "slice-2.dcm", held)
        partial = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry, replay_cache_path=cache)
        self.assertEqual(partial["overall_status"], "NOT_VERIFIED")
        self.assertFalse(cache.exists())
        shutil.move(held, dicom_dir / "slice-2.dcm")
        complete = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry, replay_cache_path=cache)
        self.assertEqual(complete["overall_status"], "VERIFIED")
        replayed = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry, replay_cache_path=cache)
        self.assertEqual(replayed["overall_status"], "NOT_VERIFIED")
        self.assertEqual(replayed["files_verified"], 0)

    def test_analysis_uses_verified_bytes_after_file_swap(self):
        target = self.root / "swap.dcm"
        shutil.copy2(self.dicom, target)
        baseline = wf.verify_then_analyze(target, self.manifest, self.registry)
        real_verify = wf.verify_instance

        def swap_after_verify(*args, **kwargs):
            outcome = real_verify(*args, **kwargs)
            shutil.copy2(self.tampered, target)
            return outcome

        with mock.patch.object(wf, "verify_instance", swap_after_verify):
            swapped = wf.verify_then_analyze(target, self.manifest, self.registry)
        self.assertEqual(swapped["cryptographic"]["status"], "VERIFIED")
        self.assertEqual(swapped["structural"], baseline["structural"])
        self.assertNotIn("modality_not_CT", swapped["structural"])

    def test_directory_analysis_uses_verified_bytes_after_file_swap(self):
        dicom_dir, manifest_dir = self.make_study(2)
        changed = dcmread(dicom_dir / "slice-1.dcm")
        changed.Modality = "MR"
        replacement = self.root / "replacement.dcm"
        dcmwrite(replacement, changed, enforce_file_format=True)
        real_verify = wf.verify_instance

        def swap_after_verify(path, *args, **kwargs):
            outcome = real_verify(path, *args, **kwargs)
            if Path(path).name == "slice-1.dcm":
                shutil.copy2(replacement, path)
            return outcome

        with mock.patch.object(wf, "verify_instance", swap_after_verify):
            result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
        self.assertEqual(result["overall_status"], "VERIFIED")
        self.assertFalse(any("modality_not_CT" in item["structural"] for item in result["results"].values()))

    def test_analysis_can_pin_expected_signer(self):
        forged = self.rewrite("forged.dcm", lambda ds: setattr(ds, "StudyDescription", "Second signer"))
        forged_manifest = self.root / "forged.manifest.json"
        save_signed_manifest(create_signed_manifest(forged, self.b_private), forged_manifest)
        result = wf.verify_then_analyze(forged, forged_manifest, self.registry, expected_signer=self.a_id)
        self.assertEqual(result["cryptographic"]["status"], "UNTRUSTED_SIGNER")

    def test_study_rejects_instance_from_other_signer(self):
        dicom_dir, manifest_dir = self.make_study()
        save_signed_manifest(create_signed_manifest(dicom_dir / "slice-1.dcm", self.b_private),
                             manifest_dir / "slice-1.dcm.manifest.json")
        result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
        self.assertEqual(result["overall_status"], "NOT_VERIFIED")
        self.assertEqual(next(iter(result["study_verification"].values()))["details"]["instances_from_other_signer"], 1)

    def test_malformed_sidecars_do_not_stop_directory_run(self):
        dicom_dir, manifest_dir = self.make_study()
        sidecar = manifest_dir / "slice-1.dcm.manifest.json"
        schema = '"schema_version":2,"canonicalization_profile":"dicom-json-v1","signature_algorithm":"Ed25519"'
        cases = ['{"manifest":[],"signature":{}}',
                 '{"manifest":{' + schema + f',"signer_id":"{self.a_id}","x":NaN' + '},"signature":{"algorithm":"Ed25519","value_base64":"AAAA"}}',
                 '{"manifest":{' + schema + ',"signer_id":[]},"signature":{"algorithm":"Ed25519","value_base64":"AAAA"}}',
                 "not json"]
        for bad in cases:
            with self.subTest(bad=bad[:30]):
                sidecar.write_text(bad, encoding="utf-8")
                result = wf.verify_directory_then_analyze(dicom_dir, manifest_dir, self.registry)
                statuses = sorted(item["cryptographic"]["status"] for item in result["results"].values())
                self.assertEqual(statuses, ["MISSING_REFERENCE", "VERIFIED", "VERIFIED"])
                self.assertEqual(result["overall_status"], "NOT_VERIFIED")

    def test_audit_records_hmac_failure(self):
        hmac_key, wrong_key = self.root / "hmac.key", self.root / "wrong.key"
        generate_hmac_key(hmac_key)
        generate_hmac_key(wrong_key)
        audit, manifest = self.root / "audit.jsonl", self.root / "hmac.manifest.json"
        quiet(cli.main, ["sign", "--dicom", str(self.dicom), "--private-key", str(self.a_private),
                         "--manifest", str(manifest), "--hmac-key", str(hmac_key), "--audit-log", str(audit)])
        code, output = quiet(cli.main, ["verify", "--dicom", str(self.dicom), "--manifest", str(manifest),
                                        "--trust-registry", str(self.registry), "--hmac-key", str(wrong_key),
                                        "--audit-log", str(audit)])
        event = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output)["final_status"], "HMAC_FAILED")
        self.assertEqual(event["outcome"], "HMAC_FAILED")

    def test_failed_hmac_does_not_consume_replay_id(self):
        hmac_key, wrong_key = self.root / "hmac2.key", self.root / "wrong2.key"
        generate_hmac_key(hmac_key)
        generate_hmac_key(wrong_key)
        audit, manifest, cache = self.root / "audit2.jsonl", self.root / "hmac2.manifest.json", self.root / "hmac-cache.json"
        quiet(cli.main, ["sign", "--dicom", str(self.dicom), "--private-key", str(self.a_private),
                         "--manifest", str(manifest), "--hmac-key", str(hmac_key), "--audit-log", str(audit)])
        verify = lambda key: quiet(cli.main, ["verify", "--dicom", str(self.dicom), "--manifest", str(manifest),
                                              "--trust-registry", str(self.registry), "--hmac-key", str(key),
                                              "--replay-cache", str(cache), "--audit-log", str(audit)])[0]
        self.assertEqual(verify(wrong_key), 2)
        self.assertFalse(cache.exists())
        self.assertEqual(verify(hmac_key), 0)
        self.assertEqual(verify(hmac_key), 2)

    def test_analyze_cli_signals_failure_and_audits(self):
        audit = self.root / "analyze-audit.jsonl"
        code, _ = quiet(cli.main, ["analyze", "--dicom", str(self.tampered), "--manifest", str(self.manifest),
                                   "--trust-registry", str(self.registry), "--output", str(self.root / "report.json"),
                                   "--audit-log", str(audit)])
        event = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(code, 2)
        self.assertEqual(event["operation"], "verify_and_analyze")
        self.assertEqual(event["outcome"], "INTEGRITY_MISMATCH")

    def test_directory_cli_end_to_end(self):
        dicom_dir = self.root / "cli-study"
        study, series = generate_uid(), generate_uid()
        for number in (1, 2):
            create_dataset(dicom_dir / f"slice-{number}.dcm", study, series, number, "PHANTOM-C")
        manifests, audit, cache = self.root / "cli-manifests", self.root / "cli-audit.jsonl", self.root / "cli-cache.json"
        sign_code, _ = quiet(cli.main, ["sign-directory", "--dicom-dir", str(dicom_dir), "--private-key", str(self.a_private),
                                        "--manifest-dir", str(manifests), "--recipient-id", "hospital-b",
                                        "--audit-log", str(audit)])
        command = ["analyze-directory", "--dicom-dir", str(dicom_dir), "--manifest-dir", str(manifests),
                   "--trust-registry", str(self.registry), "--output", str(self.root / "cli-report.json"),
                   "--expected-signer", self.a_id, "--receiver-id", "hospital-b", "--replay-cache", str(cache),
                   "--audit-log", str(audit)]
        first, _ = quiet(cli.main, command)
        replay, _ = quiet(cli.main, command)
        self.assertEqual((sign_code, first, replay), (0, 0, 2))


if __name__ == "__main__":
    unittest.main()
