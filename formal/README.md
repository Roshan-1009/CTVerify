# CTVerify formal model draft

The instance model asks whether an accept event at the pinned receiver corresponds to an earlier issue event for the same signer key, recipient, ID, and file bytes. It also checks that an honest exchange is reachable. The network is abstracted by Tamarin's public input and output facts. Signing and hashing are symbolic, and the pinned key and receiver are provisioned correctly. Signing keys remain uncompromised.

This initial theory covers the raw-file snapshot, recipient, and pinned signer binding. It abstracts canonical serialization into a typed tuple. It does not model the replay store, object versions, expiry, signed study membership, DICOM parsing, or inference. Those are separate verification targets. A proof of the draft would apply only to the stated abstraction.

Run under a supported Tamarin installation:

    tamarin-prover --prove formal/ctverify_strict_instance.spthy

Verification status: draft. Tamarin was unavailable in the Windows workspace, and WSL could not be invoked. No lemma is reported as verified.
