import json
import tempfile
import unittest
from pathlib import Path

from pydicom import dcmread, dcmwrite
from pydicom.uid import generate_uid

from scripts.make_demo_dataset import create_dataset
from src.security.canonical import (canonical_json_bytes, metadata_sha256,
                                    pixel_data_sha256, raw_file_sha256)
from src.security.manifest import (create_signed_manifest, save_signed_manifest,
                                   verify_signed_manifest)
from src.security.privacy import (decrypt_file, deidentify_research_copy,
                                  encrypt_file, generate_aes256_key)
from src.security.signing import (add_trusted_signer, generate_ed25519_keypair,
                                  generate_hmac_key, hmac_sha256, verify_hmac_sha256)
from src.security.workflow import verify_directory_then_analyze


class SignedDICOMSecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.tmp.name)
        self.dicom = self.root / "original.dcm"
        create_dataset(self.dicom, generate_uid(), generate_uid(), 1, "PHANTOM001")
        self.private = self.root / "sender-private.pem"
        self.public = self.root / "sender-public.pem"
        self.signer_id = generate_ed25519_keypair(self.private, self.public)
        self.registry = self.root / "trusted-signers.json"
        add_trusted_signer(self.registry, self.signer_id, self.public)
        self.trust = json.loads(self.registry.read_text(encoding="utf-8"))
        self.envelope = create_signed_manifest(self.dicom, self.private)

    def tearDown(self):
        self.tmp.cleanup()

    def test_unchanged_signed_dicom_verifies(self):
        result = verify_signed_manifest(self.dicom, self.envelope, self.trust)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(set(result["details"]["verified_scopes"]),
                         {"metadata_sha256", "pixel_data_sha256", "raw_file_sha256"})

    def test_metadata_and_pixel_changes_fail_content_integrity(self):
        metadata_copy = self.root / "metadata-change.dcm"
        ds = dcmread(self.dicom); ds.Modality = "MR"; dcmwrite(metadata_copy, ds, enforce_file_format=True)
        self.assertEqual(verify_signed_manifest(metadata_copy, self.envelope, self.trust)["status"], "INTEGRITY_MISMATCH")
        pixel_copy = self.root / "pixel-change.dcm"
        ds = dcmread(self.dicom); altered = bytearray(ds.PixelData); altered[100] ^= 1; ds.PixelData = bytes(altered)
        dcmwrite(pixel_copy, ds, enforce_file_format=True)
        result = verify_signed_manifest(pixel_copy, self.envelope, self.trust)
        self.assertEqual(result["status"], "INTEGRITY_MISMATCH")
        self.assertIn("pixel_data_sha256", result["details"]["digest_mismatches"])

    def test_tampered_manifest_signature_and_untrusted_key_are_rejected(self):
        changed = json.loads(json.dumps(self.envelope)); changed["manifest"]["dicom"]["study_instance_uid"] = "2.25.999"
        self.assertEqual(verify_signed_manifest(self.dicom, changed, self.trust)["status"], "INVALID_SIGNATURE")
        self.assertEqual(verify_signed_manifest(self.dicom, self.envelope, {"schema_version": 1, "signers": {}})["status"], "UNTRUSTED_SIGNER")
        self.assertEqual(verify_signed_manifest(self.dicom, self.envelope, self.trust,
                                                expected_signer="unexpected-signer")["status"], "UNTRUSTED_SIGNER")

    def test_malformed_input_is_not_reported_as_verified(self):
        malformed = self.root / "not-a-dicom.dcm"
        malformed.write_bytes(b"not a DICOM file")
        self.assertEqual(verify_signed_manifest(malformed, self.envelope, self.trust)["status"], "MALFORMED_DICOM")

    def test_reserialization_scopes_are_measured_independently(self):
        rewritten = self.root / "reserialized.dcm"
        ds = dcmread(self.dicom); ds.preamble = b"R" * 128
        dcmwrite(rewritten, ds, enforce_file_format=True)
        before, after = dcmread(self.dicom), dcmread(rewritten)
        self.assertEqual(metadata_sha256(before), metadata_sha256(after))
        self.assertEqual(pixel_data_sha256(before), pixel_data_sha256(after))
        self.assertNotEqual(raw_file_sha256(self.dicom), raw_file_sha256(rewritten))
        self.assertEqual(verify_signed_manifest(rewritten, self.envelope, self.trust)["status"], "INTEGRITY_MISMATCH")
        content_only = create_signed_manifest(self.dicom, self.private, include_raw_file_hash=False)
        self.assertEqual(verify_signed_manifest(rewritten, content_only, self.trust)["status"], "POLICY_REVIEW_REQUIRED")
        self.assertEqual(verify_signed_manifest(rewritten, content_only, self.trust,
                                                allow_content_only=True)["status"], "VERIFIED")

    def test_replay_cache_flags_second_acceptance(self):
        cache = self.root / "accepted-manifests.json"
        first = verify_signed_manifest(self.dicom, self.envelope, self.trust, replay_cache_path=cache)
        second = verify_signed_manifest(self.dicom, self.envelope, self.trust, replay_cache_path=cache)
        self.assertEqual(first["status"], "VERIFIED")
        self.assertEqual(second["status"], "POLICY_REVIEW_REQUIRED")

    def test_directory_flow_verifies_then_runs_group_consistency(self):
        dicom_root, manifest_root = self.root / "study", self.root / "manifests"
        study_uid, series_uid = generate_uid(), generate_uid()
        for index, patient in enumerate(("P-A", "P-B"), start=1):
            path = dicom_root / f"slice-{index}.dcm"
            create_dataset(path, study_uid, series_uid, index, patient)
            rel = path.relative_to(dicom_root)
            sidecar = manifest_root / (rel.as_posix() + ".manifest.json")
            save_signed_manifest(create_signed_manifest(path, self.private), sidecar)
        result = verify_directory_then_analyze(dicom_root, manifest_root, self.registry)
        self.assertEqual(result["files_discovered"], 2)
        self.assertEqual(result["files_verified"], 2)
        self.assertTrue(all(record["cryptographic"]["status"] == "VERIFIED"
                            for record in result["results"].values()))
        self.assertTrue(all(any("PatientID" in finding for finding in record["semantic"])
                            for record in result["results"].values()))

    def test_hmac_accepts_only_correct_shared_key(self):
        key = b"a" * 32
        message = canonical_json_bytes(self.envelope["manifest"])
        tag = hmac_sha256(key, message)
        self.assertTrue(verify_hmac_sha256(key, message, tag))
        self.assertFalse(verify_hmac_sha256(b"b" * 32, message, tag))
        self.assertFalse(verify_hmac_sha256(key, message + b"x", tag))

    def test_research_deidentification_writes_copy_and_preserves_relations(self):
        copy = self.root / "deid.dcm"
        original_before = self.dicom.read_bytes()
        report = deidentify_research_copy(self.dicom, copy, b"p" * 32)
        original = dcmread(self.dicom); deidentified = dcmread(copy)
        self.assertEqual(self.dicom.read_bytes(), original_before)
        self.assertNotEqual(original.PatientID, deidentified.PatientID)
        self.assertNotIn("PatientName", deidentified)
        self.assertEqual(deidentified.SOPInstanceUID, deidentified.file_meta.MediaStorageSOPInstanceUID)
        self.assertTrue(report["pixel_review_required"])

    def test_aes_gcm_round_trip_and_tamper_rejection(self):
        key_path = self.root / "aes.key"
        generate_aes256_key(key_path)
        key = key_path.read_bytes()
        encrypted = self.root / "encrypted.bin"
        recovered = self.root / "recovered.dcm"
        encrypt_file(self.dicom, encrypted, key)
        decrypt_file(encrypted, recovered, key)
        self.assertEqual(recovered.read_bytes(), self.dicom.read_bytes())
        payload = bytearray(encrypted.read_bytes()); payload[-1] ^= 1; encrypted.write_bytes(payload)
        with self.assertRaises(Exception):
            decrypt_file(encrypted, self.root / "tampered.dcm", key)


if __name__ == "__main__":
    unittest.main()
