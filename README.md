# DICOM Brain CT Integrity: Metadata + Pixel Tampering Detection (In-Depth Explainer)

> New cryptographic signing, HMAC, verification, privacy-copy, and security experiment documentation: [SECURITY_FRAMEWORK.md](SECURITY_FRAMEWORK.md).

> Paper: *Integrity Verification and Anomaly Detection for DICOM Brain CT: Controlled Tampering and Signed-Manifest Evaluation*
> Authors: Aditya Upendra Gupta, Aaditya Awasthi, Nidal, Sudhawalli
> Dataset: CPAISD (non-contrast brain CT, hyperacute ischemic stroke)

**Status:** exploratory. One dataset, a 10-study test fold, synthetic attacks, mostly single-seed runs. Not a clinical study.

---

## Table of contents

- [DICOM Brain CT Integrity: Metadata + Pixel Tampering Detection (In-Depth Explainer)](#dicom-brain-ct-integrity-metadata--pixel-tampering-detection-in-depth-explainer)
  - [Table of contents](#table-of-contents)
  - [1. The problem](#1-the-problem)
  - [2. Core idea in one paragraph](#2-core-idea-in-one-paragraph)
  - [3. Threat model](#3-threat-model)
  - [4. Dataset and splits](#4-dataset-and-splits)
  - [5. Pipeline overview](#5-pipeline-overview)
  - [6. Metadata track: detectors](#6-metadata-track-detectors)
    - [6.1 Canonical metadata SHA-256 (reference-based)](#61-canonical-metadata-sha-256-reference-based)
    - [6.2 Structural rules (reference-free)](#62-structural-rules-reference-free)
    - [6.3 Study/series consistency checker (reference-free)](#63-studyseries-consistency-checker-reference-free)
    - [6.4 Isolation Forest](#64-isolation-forest)
    - [6.5 Autoencoder](#65-autoencoder)
    - [6.6 Feature representation (important for the results)](#66-feature-representation-important-for-the-results)
    - [6.7 OR hybrid](#67-or-hybrid)
  - [7. Metadata track: attacks](#7-metadata-track-attacks)
  - [8. Metadata track: results explained](#8-metadata-track-results-explained)
    - [8.1 Macro-average (six original attacks, instance level)](#81-macro-average-six-original-attacks-instance-level)
    - [8.2 Per-attack detail (recall / FPR)](#82-per-attack-detail-recall--fpr)
    - [8.3 UID inconsistency: label granularity issue](#83-uid-inconsistency-label-granularity-issue)
    - [8.4 Same-length substitutions](#84-same-length-substitutions)
    - [8.5 Study/series consistency checker](#85-studyseries-consistency-checker)
    - [8.6 The coordinated edit (the checker's limit)](#86-the-coordinated-edit-the-checkers-limit)
    - [8.7 Feature-aware adaptive evasion](#87-feature-aware-adaptive-evasion)
    - [8.8 Optimizer comparison (AE, 40 epochs, seed 42)](#88-optimizer-comparison-ae-40-epochs-seed-42)
    - [8.9 Header-to-pixel layout checker](#89-header-to-pixel-layout-checker)
  - [9. Pixel track: detectors](#9-pixel-track-detectors)
  - [10. Pixel track: attacks](#10-pixel-track-attacks)
  - [11. Pixel track: results explained](#11-pixel-track-results-explained)
    - [11.1 Hashes](#111-hashes)
    - [11.2 No-reference baselines (negative result)](#112-no-reference-baselines-negative-result)
    - [11.3 30-epoch U-Net (held-out test, 809 images/condition)](#113-30-epoch-u-net-held-out-test-809-imagescondition)
    - [11.4 U-Net vs Attention U-Net](#114-u-net-vs-attention-u-net)
  - [12. Adversarial robustness (FGSM / PGD)](#12-adversarial-robustness-fgsm--pgd)
  - [13. Benign-change controls](#13-benign-change-controls)
  - [14. Combined metadata + pixel](#14-combined-metadata--pixel)
  - [15. Conformal calibration and seed sensitivity](#15-conformal-calibration-and-seed-sensitivity)
    - [Study-level conformal p-values](#study-level-conformal-p-values)
    - [Seed sensitivity](#seed-sensitivity)
  - [16. What the paper actually proves](#16-what-the-paper-actually-proves)
  - [17. Limitations](#17-limitations)
  - [18. Reproducibility](#18-reproducibility)
  - [19. Glossary](#19-glossary)
  - [20. Citation](#20-citation)
  - [21. Added clean-training baselines: kNN, PatchCore, and late fusion](#21-added-clean-training-baselines-knn-patchcore-and-late-fusion)
  - [22. Computer Networks extension: authenticated DICOM exchange](#22-computer-networks-extension-authenticated-dicom-exchange)

---

## 1. The problem

A DICOM file = **pixel array + metadata header** (PatientID, Modality, PixelSpacing, SOP/Study/Series UIDs, Manufacturer, ProtocolName, ...).

- Edit the header and the image looks identical, so image-only checks miss it.
- Edit a few pixels and the header is unchanged, so header-only checks miss it.

Existing work (digital signatures, reversible watermarking, PACS forensics; DICOM PS3.15) mostly needs a **trusted reference, key, or watermark**. Statistical anomaly detectors need none, but "unusual" is not the same as "malicious", and a plausible-but-wrong value may look perfectly normal.

**Research question:** how do hashing, rules, consistency checks, and unsupervised ML behave on the *same* controlled edits, and what does each one miss?

---

## 2. Core idea in one paragraph

Build a small, controlled benchmark on public CT data. Apply scripted edits to copies of the clean test files. Run every detector on every edit. Report recall and false-positive rate honestly, including where the "good" numbers are good only by construction (the hash).

The contribution is a **reproducible head-to-head comparison**, not a new algorithm. The paper explicitly does not claim first-ever DICOM tamper detection.

---

## 3. Threat model

Two separate tracks, both closed and non-adaptive:

| Track | Attacker changes | Attacker does NOT change |
|---|---|---|
| Metadata | selected header values | PixelData, reference manifest |
| Pixel | selected stored pixel values | metadata, reference |

Both hash comparators get the **exact clean per-file reference**.

Out of scope: manifest compromise, adaptive attackers (except one feature-collision test), PACS audit-log tampering, sender impersonation, clinical correctness. A mismatch means "differs from the known original", not "malicious".

---

## 4. Dataset and splits

CPAISD, official folds:

| Fold | Studies | Instances |
|---|---|---|
| Train | 92 | 8,376 |
| Validation | 10 | 980 |
| Test | 10 | 809 |

- All CT, and number of series = number of studies.
- Stroke annotations are **not used**.
- In the test fold, PatientID, PatientName, StudyDescription, SeriesDescription are **absent**. ProtocolName and InstitutionName are "Anonymized" everywhere. Manufacturer has two values.
- Consequence: the PatientID and StudyDescription attacks **insert** missing tags rather than modify existing ones.
- Pixel experiments decode PixelData only for the 809 test files and their edited copies.

---

## 5. Pipeline overview

```
Official CPAISD folds (92 / 10 / 10 studies)
        │
        ├── Metadata track ──► features ──► fit on clean train ──► calibrate on clean val
        │                                         │
        │                       controlled edits on copies of clean test
        │                                         ▼
        │                  SHA-256 | Rules | Consistency | IF | AE | OR hybrid
        │
        └── Pixel track ──► Pixel SHA | no-ref patch IF | median residual
                            | header-to-pixel layout | U-Net / Attention U-Net
                            | FGSM / PGD evasion
                                         ▼
              Study alerts + metrics + per-alert explanations
```

Evaluation rules: fit on clean train only, thresholds from clean validation only, a metadata baseline generated per clean test file, attack log gives file-level labels.

---

## 6. Metadata track: detectors

### 6.1 Canonical metadata SHA-256 (reference-based)

Serialize every main-dataset and file-meta (group 0002) element by tag, stringify values, **exclude PixelData and group-length elements**, dump as sorted compact UTF-8 JSON, then hash:

```
h_i = SHA256( JSON_canonical(M_i) )
```

A manifest maps each relative path to its clean digest and records the SOP/Study/Series UIDs. Flag if the path is missing from the manifest or the digest differs.

Not a digital signature. The manifest is built from the same clean files used as controls and is not signed, so the clean FPR is 0 **by construction**.

### 6.2 Structural rules (reference-free)

Flag:
- missing required tags (SOPClassUID, SOPInstanceUID, StudyInstanceUID, SeriesInstanceUID, Modality, Rows, Columns)
- non-CT modality
- non-positive Rows/Columns, invalid PixelSpacing, non-positive SliceThickness
- a SeriesInstanceUID tied to more than one StudyInstanceUID

The last rule is **contextual**: one bad slice makes its neighbors get flagged too (this causes the file-level label mismatch discussed later).

### 6.3 Study/series consistency checker (reference-free)

Checks that slices sharing an identifier agree.

- **Study level:** PatientID, PatientName, PatientBirthDate, PatientSex, StudyDate
- **Series level:** StudyInstanceUID, Modality, Manufacturer, ProtocolName, Rows, Columns, SamplesPerPixel, PhotometricInterpretation, BitsAllocated, BitsStored, HighBit, PixelRepresentation, PixelSpacing, ImageOrientationPatient, SliceThickness
- **Dependencies:** `HighBit = BitsStored - 1`; monochrome PhotometricInterpretation for single-sample CT

A conflict = values disagree, or a tag exists in some members and not others. Groups of fewer than two instances are skipped. Findings propagate to all group members, so evaluation is done **per study**, not per file.

### 6.4 Isolation Forest

300 trees on DictVectorizer features from clean training data, seed 42. Threshold = 1st percentile of clean validation scores. This is an instance-level tail rate, **not** a guaranteed study-level FPR, because slices cluster within studies.

### 6.5 Autoencoder

- Input: DictVectorizer + StandardScaler, fit on clean train
- Architecture: `d → 32 → 16 → 8 → 4 → 8 → 16 → 32 → d`, ReLU
- Loss: MSE, 40 epochs, batch 32, lr 1e-3
- Score: `e_i = (1/d) * ||z_i - f(z_i)||^2`
- Threshold: 99th percentile of clean validation errors

Optimizer ablation: Adam (baseline), RMSprop, AdamW (wd 1e-4), SGD (momentum 0.9). Everything else held fixed, one seed, no per-optimizer tuning.

### 6.5.1 Denoising-autoencoder comparison (run complete; exploratory)

The optional matched comparison trains the same MLP with clean targets while
adding Gaussian noise (`std=0.10`) and random feature masking (`p=0.05`) to
training inputs. It compares the standard AE and denoising AE using RMSprop,
the same folds and attack folders, and independent clean-validation thresholds.
It reports attack macro metrics, clean file/study false-positive rates, and
training curves. This comparison has been run once; outputs are in
`experiment_cpaisd/reports/denoising_autoencoder_comparison` and are not yet
included in the paper.

```powershell
python scripts/run_denoising_autoencoder_comparison.py `
  --train data/cpaisd/train `
  --validation data/cpaisd/val `
  --test data/cpaisd/test `
  --attacks experiment_cpaisd/attacks `
  --output experiment_cpaisd/reports/denoising_autoencoder_comparison
```

### 6.5.2 Mixed-type reconstruction loss (runs complete; exploratory)

This separate comparison keeps the MSE autoencoder as a baseline and tests an
equal-weight mixed objective: Smooth L1 for numeric features, categorical
cross-entropy for one-hot groups, and binary cross-entropy for field-presence
indicators. It uses the same folds and existing attack conditions and
calibrates each model on clean validation data.
The fixed-weight and Adam-bilevel comparisons have each been run once. Their
metrics and learned-weight curves are in the corresponding
`experiment_cpaisd/reports/` folders; the results are not yet included in the
paper.

```powershell
python scripts/run_mixed_autoencoder_comparison.py `
  --train data/cpaisd/train `
  --validation data/cpaisd/val `
  --test data/cpaisd/test `
  --attacks experiment_cpaisd/attacks `
  --output experiment_cpaisd/reports/mixed_autoencoder_comparison
```

For a learned mixed-loss weighting ablation, use
`scripts/run_bilevel_mixed_autoencoder_comparison.py`. The three group weights
start equal and are optimized by an Adam hypergradient update; the AE weights
remain trained with RMSprop to match the MSE baseline. Validation studies are
split so the synthetic edits used to learn weights do not overlap the clean
studies used for threshold calibration. This trains and compares MSE, fixed
mixed, and bilevel mixed models:

```powershell
python scripts/run_bilevel_mixed_autoencoder_comparison.py `
  --train data/cpaisd/train `
  --validation data/cpaisd/val `
  --test data/cpaisd/test `
  --attacks experiment_cpaisd/attacks `
  --output experiment_cpaisd/reports/bilevel_mixed_autoencoder_comparison
```

### 6.6 Feature representation (important for the results)

- Numeric: rows, columns, bit depth, slice thickness, spacing, rescale params, KVP, tube current
- Text fields (PatientID, ProtocolName, ...): **only presence and length**, never content
- Categorical: Manufacturer, Modality, PhotometricInterpretation, BodyPartExamined
- Missing or non-numeric values map to defaults, so a true zero can look like missing

### 6.7 OR hybrid

```
y_hat = hash OR rules OR IF OR AE
```

Recall can never go below any component's recall. FPR can only go up. Not a novel fusion method.

---

## 7. Metadata track: attacks

20% of test studies (2 of 10) are selected with seed 42. The same two studies are hit in every variant.

| Attack | What changes | Positives |
|---|---|---|
| PatientID | **inserts** absent tag | 177 files |
| StudyDescription | **inserts** absent tag | 177 files |
| PixelSpacing | values × 1.7 | 177 files |
| ProtocolName removal | tag deleted | 177 files |
| Modality | CT → MR | 177 files |
| UID inconsistency | one instance per selected series | **2 files** |
| ProtocolName same-length substitution | 10 chars swapped, length preserved | 177 files |
| Manufacturer same-length substitution | 7 or 18 chars swapped | 177 files |

Extra conditions:
- **One-slice edits** (study/series consistency test): PatientID, ProtocolName, Manufacturer, HighBit violation, UID and Modality positive controls, on one instance in each of two studies.
- **Coordinated edit:** ProtocolName changed in **every** instance of two studies.
- **Feature-aware evasion:** same-length ProtocolName edit chosen because the feature map ignores content.

---

## 8. Metadata track: results explained

### 8.1 Macro-average (six original attacks, instance level)

| Method | Acc | Prec | Rec | F1 | FPR |
|---|---|---|---|---|---|
| SHA-256 reference | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Structural rules | 0.818 | 0.169 | 0.333 | 0.170 | 0.036 |
| Isolation Forest | 0.837 | 0.167 | 0.089 | 0.116 | 0.000 |
| Autoencoder | 0.825 | 0.408 | 0.667 | 0.506 | 0.171 |
| OR hybrid | 0.826 | 0.512 | 1.000 | 0.635 | 0.207 |

How to read it:
- **Hash = 1.000** is expected. The reference came from the same files.
- **Rules** only catch what they encode, so Modality→MR and UID conflicts. They miss PatientID, StudyDescription, PixelSpacing (still positive and finite), and ProtocolName.
- **Isolation Forest** only partly catches PixelSpacing (about 54% of edited files). It misses everything else.
- **Autoencoder** flags all edits for four of the five whole-study attacks, but misses Modality, and has **13.8% clean file FPR** (112 of 809 clean files).
- **Hybrid recall 1.000** is inherited from the hash. FPR rises because the AE's false alarms are OR-ed in. This is not evidence that ML improves integrity.

### 8.2 Per-attack detail (recall / FPR)

| Attack | n+ | SHA-256 | Rules | IF | AE | OR |
|---|---|---|---|---|---|---|
| PatientID | 177 | 1.00/0.00 | 0.00/0.00 | 0.00/0.00 | 1.00/0.18 | 1.00/0.18 |
| StudyDescription | 177 | 1.00/0.00 | 0.00/0.00 | 0.00/0.00 | 1.00/0.18 | 1.00/0.18 |
| PixelSpacing | 177 | 1.00/0.00 | 0.00/0.00 | 0.54/0.00 | 1.00/0.18 | 1.00/0.18 |
| UID inconsistency | 2 | 1.00/0.00 | 1.00/0.22 | 0.00/0.00 | 0.00/0.14 | 1.00/0.36 |
| Protocol removal | 177 | 1.00/0.00 | 0.00/0.00 | 0.00/0.00 | 1.00/0.18 | 1.00/0.18 |
| Modality change | 177 | 1.00/0.00 | 1.00/0.00 | 0.00/0.00 | 0.00/0.18 | 1.00/0.18 |

**Why the AE catches PatientID/StudyDescription:** these tags were absent everywhere, so inserting them changes the presence/length features. That's a *presence shift*, not semantic detection.

### 8.3 UID inconsistency: label granularity issue

The rule flags **all** slices in the inconsistent series (177 files), but only 2 edited files are labeled positive, giving 175 "false positives" at file level. At **series level** (the right unit) the rule gets recall 1.000, FPR 0.000 (2/2 series, 0/8 false).

### 8.4 Same-length substitutions

| Attack | SHA-256 | IF | AE |
|---|---|---|---|
| ProtocolName | 1.00/0.00 | 0.00/0.00 | 0.00/0.177 |
| Manufacturer | 1.00/0.00 | 0.00/0.00 | 1.00/0.177 |

AE catches Manufacturer (it's a categorical feature) but not ProtocolName (content ignored). Descriptive for these specific substitutions only.

### 8.5 Study/series consistency checker

On clean data: **zero findings**. On six one-slice edits (2 positive studies, 8 negative each):

| Condition | Structural rules | Consistency |
|---|---|---|
| PatientID | 0.00/0.00 | 1.00/0.00 |
| ProtocolName | 0.00/0.00 | 1.00/0.00 |
| Manufacturer | 0.00/0.00 | 1.00/0.00 |
| HighBit violation | 0.00/0.00 | 1.00/0.00 |
| Study UID (control) | 1.00/0.00 | 1.00/0.00 |
| Modality (control) | 1.00/0.00 | 1.00/0.00 |

It marks 151 to 177 instances per condition because findings propagate to peers. Those are not file-level precision numbers, and the checker can't say *which* slice is the odd one out.

### 8.6 The coordinated edit (the checker's limit)

| Condition | Consistency | Structural | Metadata hash |
|---|---|---|---|
| One instance per study | 2/2 | 0/2 | 2/2 |
| **Every** instance in study | **0/2** | 0/2 | 2/2 |

If every slice carries the same wrong value, the group is internally consistent and the checker sees nothing. Only the reference hash catches it. This is the paper's cleanest illustration of **internal consistency vs. reference verification**.

### 8.7 Feature-aware adaptive evasion

Same-length ProtocolName substitution on 177 files:
- Feature vectors are **exactly unchanged**
- IF score and AE error change by exactly **0**
- Rules, IF, AE: **0/177**. Hash: **177/177**

Lesson: an unsupervised model cannot detect information discarded before it sees the input. This is a representation blind spot, not a clever optimizer.

### 8.8 Optimizer comparison (AE, 40 epochs, seed 42)

| Optimizer | Attack F1 (file / study) | Attack FPR (file / study) | Clean FPR (file / study) |
|---|---|---|---|
| Adam | 0.506 / 0.533 | 0.171 / 0.125 | 0.1384 / 0.100 |
| RMSprop | **0.831 / 0.667** | 0.002 / 0.125 | **0.0012** / 0.100 |
| AdamW | 0.551 / 0.617 | 0.178 / 0.125 | 0.1446 / 0.100 |
| SGD (m=0.9) | 0.611 / 0.533 | 0.049 / 0.125 | 0.0396 / 0.100 |

RMSprop wins in this fixed setup, but: one seed, shared untuned lr, ten test studies. All four optimizers flag 1 of 10 clean studies. A candidate result, not an optimizer recommendation.

### 8.9 Header-to-pixel layout checker

Reference-free. Checks that Rows/Columns/SamplesPerPixel/NumberOfFrames/bit-depth are valid, that uncompressed PixelData length matches the declared dimensions, that decoded shape matches the header, and that decoded values fit BitsStored/PixelRepresentation.

Results: 0 findings on 809 clean files; **177/177** Rows-mismatch and **177/177** Columns-mismatch caught; 0 flags on pixel-only edits (layout stays valid). These contradictions are deliberately obvious. It doesn't catch plausible-but-wrong metadata that still fits the payload.

---

## 9. Pixel track: detectors

1. **Pixel SHA-256** over raw PixelData vs a per-file clean digest (exact-reference control). Metadata SHA-256 serves as a **negative control** (it excludes PixelData).
2. **No-reference patch Isolation Forest:** robust scaling by the 1st/99th percentile, non-overlapping 16×16 patches, 6 features (mean, std, horizontal/vertical gradients, high-deviation fraction, log-std). 100 trees on 50,000 clean patches from 2,084 training slices. Localization is coarse (16×16 blocks).
3. **Median-filter residual:** `R = |I - Median5x5(I)|`, pixel threshold at the 99.9th percentile of clean validation residuals, image threshold at the 99th percentile of image scores.
4. **Compact U-Net** (supervised, no reference):
   - base width 8, widths 8-16-32-64-128 down, 64-32-16-8 up, skip connections, 1×1 conv to logits
   - input robustly normalized and resized to 256×256
   - loss = 0.5 weighted BCE (pos weight 12) + 0.5 soft Dice, AdamW lr 1e-3, wd 1e-4
   - trained on synthetic localized patches + sparse low-amplitude edits; **copy-move withheld** as an unseen family
   - thresholds from clean validation: pixel = 99.5th percentile, image = 99th percentile of image-max probabilities
5. **Attention U-Net:** same everything, plus attention gates on each skip connection (encoder skip weighted by a sigmoid map from decoder context).

---

## 10. Pixel track: attacks

All edits are made in the **stored integer domain** (no RescaleSlope/Intercept applied, so not HU).

| Attack | Default parameters | Files |
|---|---|---|
| Localized patch | +32 on central 5×5, every slice of selected studies | 177 |
| Low-amplitude | +1 on random 1% of pixels per slice | 177 |
| Slice inconsistency | +8 on central 5×5, one slice per selected series | 2 |

Severity sweep:

| Family | Subtle | Default | Stronger |
|---|---|---|---|
| Patch | 3×3, +1 | 5×5, +32 | 9×9, +32 |
| Sparse | 0.1%, +1 | 1%, +1 | 5%, +2 |
| Slice | 3×3, +1 | 5×5, +8 | 9×9, +32 |

The difference map is `D(u,v) = |P'(u,v) - P(u,v)|`. A mask-vs-`D>0` comparison gives IoU 1.000 **by construction** because both come from the same script. It validates bookkeeping, not localization ability.

---

## 11. Pixel track: results explained

### 11.1 Hashes

| Attack | Pixel SHA-256 | Metadata SHA-256 |
|---|---|---|
| Localized patch | 1.00/0.00 | 0.00/0.00 |
| Low-amplitude | 1.00/0.00 | 0.00/0.00 |
| Slice inconsistency | 1.00/0.00 | 0.00/0.00 |

Pixel hash catches everything at all nine attack/severity combos. Any changed byte changes the digest. Metadata hash catching nothing is the expected negative control. This says nothing about the smallest *clinically meaningful* change.

### 11.2 No-reference baselines (negative result)

- **Patch IF:** image-level FPR 0.0111; detected **no attacked image** at its threshold; localization recall about 0 (IoU 0.0034 on low-amplitude).
- **Median residual:** clean image alarm 8.78%; recall 6.21% (patch), 5.65% (low-amplitude), 0% (slice); IoU about 0.

Tiny edits are statistically invisible against normal CT texture. A no-reference detector is much harder than exact-reference hashing.

### 11.3 30-epoch U-Net (held-out test, 809 images/condition)

| Family | n+ | AUROC | Img recall | Img FPR | Img F1 | Px precision | Px recall | Px IoU |
|---|---|---|---|---|---|---|---|---|
| Patch | 236 | 0.984 | 0.996 | 0.030 | 0.963 | 0.376 | 0.999 | 0.376 |
| Sparse | 152 | 0.986 | 1.000 | 0.027 | 0.944 | 0.339 | 0.964 | 0.335 |
| Copy-move (unseen) | 139 | 0.972 | 0.885 | 0.024 | 0.885 | 0.424 | 0.812 | 0.386 |

- Epoch 30 chosen by validation macro IoU (0.9254). Validation numbers are **not independent** (used for selection).
- Image-level numbers look great, but **pixel precision is only 0.34 to 0.42** (lots of highlighted pixels outside the true mask) and **study-level FPR is 0.50 to 0.625** (recall 1.0).
- Clean image FPR: 2.84%.
- The 4-epoch pilot is superseded (recall ≤ 0.114).

### 11.4 U-Net vs Attention U-Net

| Metric | U-Net | Attention |
|---|---|---|
| Best epoch | 30 | 28 |
| Val IoU | 0.9254 | 0.9322 |
| Clean image FPR | 2.84% | 3.83% |
| Patch IoU | 0.376 | 0.397 |
| Sparse IoU | 0.335 | **0.214** |
| Copy-move IoU | 0.386 | 0.420 |
| Image F1 | higher on all three families | lower on all three |

Study-level results are identical. Net effect: **family-dependent, no overall win for attention**.

---

## 12. Adversarial robustness (FGSM / PGD)

Setup: the trained U-Net, no retraining. Targeted white-box attack minimizing BCE against an **all-clean mask** to suppress the tamper alarm. Budget `L_inf = 0.02` on the normalized [0,1] input. PGD: 10 steps, step 0.004, random start. Thresholds stay fixed from clean validation. Only edited positive images are attacked. Attack success rate (ASR) = fraction of baseline-detected images that fall below the alarm threshold afterward.

| Family | Baseline recall | FGSM recall | FGSM ASR | PGD recall | PGD ASR | PGD IoU |
|---|---|---|---|---|---|---|
| Patch | 0.996 | 0.877 | 0.119 | 0.097 | 0.902 | 0.081 |
| Sparse | 1.000 | 0.217 | 0.783 | **0.000** | 1.000 | 0.117 |
| Copy-move | 0.885 | 0.547 | 0.390 | 0.173 | 0.805 | 0.057 |

Meaning: this checkpoint is highly evadable by a small targeted white-box perturbation. Not shown: that such perturbations are clinically plausible, survive serialization back into DICOM/HU, or transfer to other detectors. An exact-reference PixelData hash would still catch the serialized change.

---

## 13. Benign-change controls

Three proxies applied to all 809 clean files (PixelData and non-target clinical attributes verified unchanged):

1. File-meta `ImplementationVersionName` update
2. Limited privacy-cleanup proxy (de-id markers)
3. Re-serialization through pydicom, no attribute edit

| Method | File-meta update | Privacy cleanup | Re-serialize |
|---|---|---|---|
| Metadata SHA-256 | 1.000 | 1.000 | **0.000** |
| Structural rules | 0 | 0 | 0 |
| Isolation Forest | 0 | 0 | 0 |
| Autoencoder | 0.138 (0 new) | 0.138 (0 new) | 0.138 (0 new) |
| OR hybrid | 1.000 (0.861 new) | 1.000 (0.861 new) | 0.138 (0 new) |

The strict hash answers "did a covered value change?", **not** "is it malicious?". It alarms on harmless workflow edits. The AE's 112 alarms are the same pre-existing clean-file alarms, nothing newly triggered. Takeaway: test on real PACS exports before treating a hash mismatch as a security alarm.

---

## 14. Combined metadata + pixel

Corrected run: metadata edits seed 42, pixel edits seed 44, giving **disjoint** study pairs. Per condition: 177 metadata files + 133 pixel files = **310 unique positives**.

| Detector | Recall | FPR |
|---|---|---|
| Metadata SHA-256 | 0.571 | 0.000 |
| PixelData SHA-256 | 0.429 | 0.000 |
| **OR of hashes** | **1.000** | 0.000 |

(Initial run used the same seed for both, so both hashes hit the same studies. That was fixed.) This shows two exact-reference checks cover complementary attack surfaces. It is not learned fusion and doesn't work without a trusted reference.

---

## 15. Conformal calibration and seed sensitivity

### Study-level conformal p-values

Per-study score = max anomaly severity over its slices (`-s_IF` for IF, reconstruction error for AE).

```
p(q) = (1 + sum_j 1[q_j^(v) >= q]) / (n_v + 1)
```

Alert when `p(q) <= alpha`. With only **10 validation studies**, the smallest possible p is 1/11 ≈ 0.091.

| alpha | Result |
|---|---|
| 0.05 | No alert can ever fire |
| 0.10 | Clean-test study FPR: IF 0.000, AE 0.100 |
| 0.20 | Clean-test study FPR jumps: IF 0.300, AE 0.600 |

At alpha = 0.10, IF detects PixelSpacing; AE detects Manufacturer and PixelSpacing but misses Modality and UID events. The nominal level gives **no population-level guarantee**.

### Seed sensitivity

Four attacks repeated with study-selection seeds 42/43/44, with a 2,000-resample stratified study/series bootstrap. AE study FPR is about 0.125 with CIs such as [0.000, 0.250]. Recall intervals are degenerate (0 or 1), since each detector makes the same decision on both attacked studies. The seeds reuse the same ten studies, so this does **not** give independent-site evidence.

---

## 16. What the paper actually proves

**Supported:**
- A reference hash catches every changed file by construction.
- Rules and consistency checks catch only the violations they encode.
- Consistency checks miss coordinated, internally consistent edits.
- Learned detectors can't see anything the feature map drops (ProtocolName content).
- OR-fusion inherits the hash's recall and the ML models' false alarms.
- Simple no-reference pixel detectors fail on tiny scripted edits.
- A supervised U-Net localizes well on its own synthetic edits but has low pixel precision, high study-level FPR, and is easy to evade with a small white-box PGD attack.
- Strict hashes alarm on benign metadata rewrites.

**Not supported:**
- Clinical utility, deployed sender attribution, or real-world tamper detection
- Cross-site / external generalization
- Robustness to broad adaptive attackers
- That RMSprop is generally best, or that attention helps
- That ML improves on cryptographic verification

---

## 17. Limitations

1. Single-center, 112-study cohort; test fold = **10 studies**
2. Synthetic scripted attacks; benign-like controls are proxies, not named PACS outputs
3. Reference manifest built from the same clean test files, not signed
4. Series/file label mismatch for the UID attack (reported at series level instead)
5. Text fields reduced to presence/length; unknown categories may be dropped by the vectorizer
6. Optimizer comparison: one seed, shared untuned lr
7. Semantic checker assumes within-study stability; mixed-vendor or re-export variation could break it; findings identify a group, not a slice
8. Coordinated-edit test: two studies, one field, one substitution
9. Software versions and hardware not fully recorded
10. Pixel track: capped 512-image training set, resized 256×256 grid, edits not in HU, adversarial examples not serialized to DICOM, clean-image adversarial FPR not measured
11. Seed/bootstrap analysis reuses the same ten studies
12. Conformal calibration limited to 10 validation studies
13. Layout attacks are obvious (Rows/Columns); plausible-but-consistent edits untested
14. Explanations (AE feature contributions) are diagnostic, not causal; a hash can't say which tag changed
15. No clinical or forensic validation; sender authentication is demonstrated only in a local signed-manifest prototype, not a deployed workflow

---

## 18. Reproducibility

Common settings: seed 42, 20% study attack fraction, CPU runs.

| Experiment | Key settings |
|---|---|
| Metadata + optimizer comparison | all 8,376 / 980 / 809 files, 40 epochs, batch 32, lr 1e-3 |
| 30-epoch U-Net | 512 capped train images, batch 8, 256×256, about 49 min (2,946 s), epoch 30 selected |
| Attention U-Net | same setup, epoch 28 selected |
| FGSM/PGD | eps 0.02, 10 PGD steps, step 0.004; 236 / 152 / 139 positive images |
| Combined run | metadata seed 42, pixel seed 44, 310 unique positives |

Output folders named in the paper:

```
experiment_cpaisd_semantic_consistency/
experiment_cpaisd_coordinated_metadata/
experiment_cpaisd_supervised_localization/              # 4-epoch pilot (superseded)
experiment_cpaisd_supervised_localization_512_30ep_*/   # completed U-Net run
experiment_cpaisd_attention_unet_512_30ep/
experiment_cpaisd_unet_fgsm_pgd/
```

> A public, versioned code archive has **not yet been deposited**. An external-evaluation script exists but no external cohort was run (CQ500 is a candidate). Dataset: [Zenodo 10.5281/zenodo.10892316](https://doi.org/10.5281/zenodo.10892316); follow its license and citation terms.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| DICOM | Standard format combining image pixels and a metadata header |
| SOP / Study / Series UID | Unique IDs for an instance, a study, and a series |
| PixelData | The raw pixel payload element (7FE0,0010) |
| FPR | False positive rate = FP / (FP + TN) |
| Recall | TP / (TP + FN) |
| Macro average | Unweighted mean across attack types |
| Isolation Forest | Unsupervised outlier detector based on random partitioning |
| Autoencoder | Network trained to reconstruct clean inputs; high error = anomaly |
| Conformal p-value | Rank of a score among clean calibration scores |
| FGSM / PGD | One-step / iterative gradient-sign adversarial attacks |
| ASR | Attack success rate: baseline-detected images that become undetected |
| IoU / Dice | Overlap measures between predicted and true masks |
| Feature collision | An edit that leaves the model's input vector unchanged |
| HU | Hounsfield units (CT intensity scale) |

---

## 20. Citation

```bibtex
@misc{gupta_dicom_integrity,
  title  = {Integrity Verification and Anomaly Detection for DICOM Brain CT: Controlled Tampering and Signed-Manifest Evaluation},
  author = {Gupta, Aditya Upendra and Awasthi, Aaditya and Nidal and Sudhawalli},
  note   = {Manuscript}
}
```

Dataset: D. Umerenkov, S. Kudin, M. Peksheva, D. Pavlov, "Core-penumbra hyperacute ischemic stroke dataset," *Scientific Data*, vol. 12, 707, 2025, doi:10.1038/s41597-025-05000-0.

---

## 21. Added clean-training baselines: kNN, PatchCore, and late fusion

These are independent extensions to the existing pipeline. They do not replace
the Isolation Forest, metadata autoencoder, no-reference pixel baselines, or
U-Net experiments. Do not report results until the scripts have been run and
their output files reviewed.

### Metadata kNN

`src/knn_anomaly.py` uses the existing metadata feature extractor and scores an
instance by its mean distance to the `k` nearest clean training instances.
Larger scores indicate greater anomaly. The threshold is calibrated from clean
validation scores. Outputs include file- and study-level metrics by attack.

```powershell
python scripts/run_knn_metadata_experiment.py `
  --train data/cpaisd/train --validation data/cpaisd/val --test data/cpaisd/test `
  --attacks-root experiment_cpaisd_combined_independent/combined_attacks `
  --labels-csv experiment_cpaisd_combined_independent/combined_predictions.csv `
  --output experiment_cpaisd_knn_metadata
```

### PatchCore-style pixel baseline

`src/patchcore.py` uses fixed ResNet-18 layer2/layer3 features, a clean-training
patch memory bank, and nearest-neighbor distances for image and spatial anomaly
scores. To bound memory, this implementation uses deterministic random patch
subsampling; it does not implement the original k-center coreset selection. By
default, it transfers ImageNet weights to grayscale CT by repeating the image
across three channels and applying ImageNet normalization. This transfer
assumption must be reported and evaluated. `torchvision` is required for this
optional experiment; install a build matching the PyTorch version and CPU/CUDA
setup on the machine:

```powershell
python -c "import torch; print(torch.__version__)"
# Install the matching torchvision wheel from the official PyTorch selector.
python scripts/run_patchcore_experiment.py `
  --train data/cpaisd/train --validation data/cpaisd/val --test data/cpaisd/test `
  --attacks-root experiment_cpaisd_combined_independent/combined_attacks `
  --labels-csv experiment_cpaisd_combined_independent/combined_predictions.csv `
  --output experiment_cpaisd_patchcore
```

The `--weights none` option is for pipeline debugging only: it uses randomly
initialized frozen features and is not the ImageNet PatchCore baseline.
Training and inference may be slow on CPU.

### Late fusion

`scripts/run_late_fusion.py` joins branch predictions by attack and relative
file path, and errors if files or labels do not match. It reports each branch,
equal-weight score fusion, and OR/AND decision fusion. The score weights are
fixed; score scaling and the score-fusion cutoff use clean validation scores
only. Pass the existing `combined_predictions.csv` to both branch runners so
metadata, pixel, and union labels stay separate. Run both branches against
the same paired attack folder:

```powershell
python scripts/run_late_fusion.py `
  --metadata-results experiment_cpaisd_knn_metadata `
  --patchcore-results experiment_cpaisd_patchcore `
  --output experiment_cpaisd_late_fusion
```

These extensions use the same ten-study test fold. Report study-level results
alongside file-level metrics, preserve the clean originals, and describe alarms
as anomaly signals rather than remediation or proof of malicious intent.

---

## 22. Computer Networks extension: authenticated DICOM exchange

### Motivation and networking concepts

DICOM transfers carry both image pixels and metadata. A transfer can arrive with changed bytes, an altered header, or a replaced reference digest. A hash can show that content differs from a stored digest, but it cannot show who created that digest. This extension adds an authenticated reference: a sender signs a manifest, and a receiver verifies the signature and DICOM content before running the existing integrity and anomaly checks.

The prototype connects the project to several Computer Networks security concepts:

| Concept | Implementation in this project |
|---|---|
| Message integrity | SHA-256 digests cover selected DICOM content scopes. |
| Sender-key authentication | Ed25519 signs the manifest; the receiver verifies it using a locally trusted public key. |
| Trust establishment | The public key is registered out of band. A key received alongside a file is not trusted automatically. |
| Shared-secret authentication | Optional HMAC-SHA256 verifies a tag from a shared-key holder, but cannot identify which holder created it. |
| Freshness and replay handling | A local cache flags manifest identifiers that have already been accepted. |
| Confidentiality | Network confidentiality is not implemented. AES-GCM is provided only as an at-rest file wrapper. |
| Auditability | JSONL audit events record operation outcomes and digests without raw patient identifiers. |
| Anomaly detection | Existing rules and saved AI models run after cryptographic verification; their alerts remain separate from signature status. |

The cryptographic layer and anomaly detectors answer different questions. A valid signed manifest indicates that the object matches content authenticated by a trusted key. An anomaly alert indicates that verified content looks unusual under a detector. Neither result establishes clinical correctness or malicious intent.

### Sender–receiver workflow

```text
SENDER                                               RECEIVER
  │                                                     │
  ├─ Read DICOM and calculate configured digests       │
  ├─ Build manifest with identifiers and manifest ID  │
  ├─ Sign manifest with Ed25519 private key            │
  └─ Send DICOM object + signed JSON sidecar ─────────►│
                                                        ├─ Find signer in local trust registry
                                                        ├─ Verify Ed25519 signature
                                                        ├─ Compare manifest identifiers/digests
                                                        ├─ Check replay cache (if enabled)
                                                        │
                                               ┌────────┴────────┐
                                               │                 │
                                           VERIFIED        REJECT/REVIEW
                                               │           invalid, missing,
                                               ▼           untrusted, changed,
                                    Existing integrity and  or replayed manifest
                                    AI analysis pipeline
```

The receiver establishes signer trust and validates the signature before treating the manifest digests as authoritative. If the signature, signer trust, reference, identifiers, or enabled content digests fail, the object receives a failure or review status. A successful AI result cannot override a cryptographic failure, and a successful cryptographic check does not suppress an AI anomaly alert.

The prototype also includes a local replay cache, signed study manifests, privacy-minimized audit records, and a limited research-copy transformation followed by fresh signing. HMAC-SHA256 is an optional shared-secret integrity/authentication mechanism. AES-GCM is available for files at rest; it does **not** encrypt a network connection. Manifest schema 2 hardens this workflow against slice omission, rollback, replay, and file swaps after verification; see [Workflow hardening](SECURITY_FRAMEWORK.md#workflow-hardening-manifest-schema-2). The full-fold results below were produced with schema 2.

### Manifest and verification model

The signed JSON sidecar records a schema version, manifest type and identifier, creation and expiry times, optional recipient, object version, DICOM identifiers, digest scopes, and signer key identifier. The sender signs a canonical representation of the manifest. Depending on the selected profile, it includes:

- **Canonical metadata SHA-256:** recursively serialized DICOM metadata and file-meta values; PixelData and group-length elements are excluded.
- **PixelData SHA-256:** the raw PixelData payload.
- **Raw-file SHA-256:** every serialized DICOM file byte, including preamble and representation details. It is required by default; content-only verification needs an explicit Python API opt-in.

These scopes answer different questions. A covered metadata edit changes the metadata digest; a pixel edit changes the PixelData digest; a byte-level reserialization may change the raw-file digest even when parsed metadata and PixelData remain the same. The receiver reports the failed scope instead of treating every mismatch as the same event.

The receiver's trust registry stores public keys accepted locally. Keys must be authenticated out of band; receiving a public key beside a DICOM file is not a safe trust-establishment process. In an institution, key issuance, custody, rotation, revocation, and mapping key IDs to organizational identities require an operational policy. The local registry demonstrates trust checking but is not a full public-key infrastructure.

### Full-fold security experiment

The full-fold runner tested each scenario against all **809 test files** using manifest schema 2 at commit `39fb4e8`, producing 10,517 file-scenario observations. It uses temporary experiment keys and temporary edited copies. It does not train models and does not modify the original CPAISD files. The results are in [`reports/crypto_security_full_schema2`](reports/crypto_security_full_schema2/SECURITY_EXPERIMENT_REPORT.md); `reports/crypto_security_full` is kept as the schema 1 baseline.

These 13 scenarios do not exercise the workflow attacks fixed in schema 2. Those attacks are covered on synthetic DICOM files by [`tests/test_security_workflow_attacks.py`](tests/test_security_workflow_attacks.py).

| Security scenario | Expected receiver behavior | Observed result |
|---|---|---|
| Unchanged signed DICOM object | Accept trusted signature and matching digests | `VERIFIED`, 809/809 |
| Metadata value edited after signing | Detect metadata-digest mismatch | `INTEGRITY_MISMATCH`, 809/809 |
| PixelData byte flipped after signing | Detect pixel-digest mismatch | `INTEGRITY_MISMATCH`, 809/809 |
| One decoded pixel changed after signing | Detect pixel-digest mismatch | `INTEGRITY_MISMATCH`, 809/809; decoded comparison confirmed one changed pixel per file |
| Manifest digest altered without a new valid signature | Reject altered signed content | `INVALID_SIGNATURE`, 809/809 |
| Signature bytes forged | Reject invalid signature | `INVALID_SIGNATURE`, 809/809 |
| Signer key not in trust registry | Reject unknown signer | `UNTRUSTED_SIGNER`, 809/809 |
| Signed manifest absent | Do not treat object as reference-verified | `MISSING_REFERENCE`, 809/809 |
| Incorrect HMAC key supplied | Reject shared-secret authentication tag | `INVALID`, 809/809 |
| Previously accepted manifest ID seen again | Flag possible replay for policy review | `POLICY_REVIEW_REQUIRED`, 809/809 |
| DICOM preamble changed, parsed content retained | Raw-file scope detects byte change; content-only scope verifies only with explicit opt-in | `INTEGRITY_MISMATCH`, 809/809; opted-in content-only `VERIFIED`, 809/809 |
| Attacker creates a matching unsigned digest reference | Demonstrate that digest alone is not authenticated | Digest-only control accepts, 809/809 |
| Authorized research-copy transform, then fresh signing | Verify transformed object against its new signature | `VERIFIED`, 809/809 |

The unsigned-reference result is a deliberate negative control, not a failure of the Ed25519 verifier: if an attacker can replace both the file and its unsigned hash reference, those values can match. A signature binds the manifest to a signing key; the receiver's trust policy determines whether that key is accepted. The key ID by itself does not establish a real person's or hospital's identity.

#### Timing and manifest overhead

| Measurement | Mean observed value |
|---|---:|
| Create manifest and sign it | 3.08 ms/file |
| End-to-end verification | 2.75 ms/file |
| Ed25519 signing operation | 0.047 ms |
| Ed25519 verification operation | 0.097 ms |
| Signed envelope size | 971.4 bytes |
| Envelope / serialized DICOM size | 0.1843% |

These timing values come from one local machine and measure file processing. They are not network latency, throughput, or a deployment benchmark, and they are not comparable with the schema 1 baseline, which ran on a different machine. The reserialization scenario shows the effect of digest policy: a raw-file digest detects every serialized-byte change, while canonical metadata plus PixelData can remain verified after a preamble-only change only when a caller explicitly allows content-only verification.

### Running the security experiment

Run all scenarios on the CPAISD test fold (no training):

```powershell
python scripts/run_crypto_security_experiment.py `
  --input data\cpaisd\test `
  --output reports\crypto_security_run
```

For a quicker smoke check, add `--limit 10`. The runner writes scenario and performance CSVs, a JSON summary, a Markdown report, and a timing chart to the selected output directory. Temporary experiment keys and edited file copies are removed when the run exits.

Run the automated tests with:

```powershell
pytest
```

The [security framework guide](SECURITY_FRAMEWORK.md) contains command examples for generating local test keys, adding a public key to a trust registry, signing and verifying one DICOM object, analyzing a directory, transforming a verified research copy, and encrypting a file at rest. Keep private keys outside the repository; never use temporary experiment keys for real medical records.

### Scope and limitations

This is an **application-layer secure-file-exchange prototype**, not a deployed DICOM network service. The experiment reads local files and simulates sender and receiver security scenarios. It does not open a DICOM association or transmit the test objects across a network. It does not measure TCP/TLS behavior, packet loss, throughput, concurrent sessions, or end-to-end network latency.

The prototype uses signed JSON sidecars; it does not implement DICOM-native Digital Signatures. It does not provide transport encryption: the AES-GCM helper is an at-rest file wrapper and does not replace TLS. The local replay cache is not distributed replay prevention. Experiment keys are temporary and do not identify a hospital sender. The trust registry is not an institutional PKI. The limited de-identification helper does not implement the complete DICOM PS3.15 confidentiality profile or perform OCR-based pixel cleaning. Clinical correctness, external generalization, and production readiness are not established.

The Computer Networks contribution is therefore a design and file-level prototype demonstrating content integrity, signer-key authentication, trust checking, and a replay check around DICOM objects. A complete network evaluation would connect separate sender and receiver processes over a controlled link, use authenticated encrypted transport, measure throughput and end-to-end latency, exercise connection failures and replay across receiver instances, and use a managed key lifecycle.

See [SECURITY_FRAMEWORK.md](SECURITY_FRAMEWORK.md) for architecture and implementation details, and [the paper](experiment_cpaisd/paper/main.tex) for the formal methods and results.
