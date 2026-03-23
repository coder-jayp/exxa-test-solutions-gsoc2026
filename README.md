# exxa-test-solutions-gsoc2026
GSoC 2026 ML4Sci EXXA — Test solutions for Equivariant Vision Networks (EXXA3) and Foundation Models for Exoplanet Characterization (EXXA4)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSoC](https://img.shields.io/badge/GSoC-2026-orange?logo=google)](https://summerofcode.withgoogle.com)
[![ML4Sci](https://img.shields.io/badge/ML4Sci-EXXA-purple)](https://ml4sci.org)

**Applicant:** Jay Sureshkumar Prajapati  
**Email:** jayp222001@gmail.com  
**GitHub:** [coder-jayp](https://github.com/coder-jayp)  
**Projects:** Equivariant Vision Networks for Predicting Planetary Systems' Architectures (EXXA3) · Foundation Models for Exoplanet Characterization (EXXA4)
**Organization:** ML4Sci · EXXA

| Resource | General Test | Image-Based Test | Sequential Test |
|----------|-------------|-----------------|----------------|
| Colab Notebook | [Open in Colab](https://colab.research.google.com/drive/1fIppebJ2obFtbncZxd8J6cDr_O4kDLa8?usp=sharing) | [Open in Colab](https://colab.research.google.com/drive/1IwcIekgtpDyxjjDNQKbS8ecmciD_KIaP?usp=sharing) | [Open in Colab](https://colab.research.google.com/drive/11IPgAqw-Rg9kupgENxbbqsvC9PA19_ps?usp=sharing) |
| Google Drive Inputs | — | — | [View Inputs](https://drive.google.com/drive/folders/19OQw09IPMpGzO0ny8qVUpAPCDNk7_jJZ?usp=drive_link) |
| Google Drive Outputs | [View Outputs](https://drive.google.com/drive/folders/1ck1IyUC4WgEluBtMInOQbNYiq7OI0xFk?usp=sharing) | [View Outputs](https://drive.google.com/drive/folders/13FJm4BekkhtgUcbUh0kejgzCd3XULzLI?usp=sharing) | [View Outputs](https://drive.google.com/drive/folders/1iVgNq7yy5pFB1UJAxN49tMmIpl5BJDfJ?usp=drive_link) |

## Repository Structure

```
exxa-test-solutions-gsoc2026/
├── EXXA_General_Test.ipynb        # Unsupervised disk morphology clustering
├── EXXA_Image_Based_Test.ipynb    # Attention autoencoder for disk reconstruction
├── EXXA_Sequential_Test.ipynb     # Transit light curve binary classifier (EXXA4)
├── README.md
└── LICENSE
```

---

## Test 1 — General Test: Unsupervised Disk Clustering

**Notebook:** `EXXA_General_Test.ipynb`

### Objective

Automatically discover morphological groups in 150 synthetic ALMA 1250 μm continuum
observations using unsupervised machine learning — with no labels provided.

### Pipeline Overview

```
FITS Loading → Geometry Correction → Augmentation → Feature Extraction → UMAP → K-Means → Physics Labels
  150 disks      ellipse de-proj      150→600 imgs     27D vector          5D      K=3       auto-derived
```

Six-stage fully automated pipeline. Trained models are saved to disk — classifies
withheld data without retraining.

### Key Design Decisions

| Decision | Implementation | Why |
|----------|---------------|-----|
| De-projection | Image moment ellipse fitting via `cv2.moments` | Removes inclination confound — without this, clustering separates by viewing angle, not physics |
| Feature space | 27D physics-motivated vector | Targets rings, gaps, FFT periodicity; excludes disk mass proxies (total flux, peak brightness) |
| Dimensionality reduction | Dual UMAP (5D clustering + 2D viz) | Preserves non-linear morphology manifold; better cluster separation than PCA |
| Cluster count | K=3 with domain-knowledge override | Maps directly to three physically distinct morphological classes; auto-fallback if silhouette < 0.35 |
| Augmentation | Random rotation (0–360°) + horizontal flip | Physically valid — disk emission is rotationally symmetric |
| Physics labels | Derived from centroid feature values | Fully data-driven; no manual annotation required |

### Feature Vector — 27 Dimensions

| Group | Features | Physical Motivation |
|-------|----------|-------------------|
| Substructure | Ring count, gap count | Direct signatures of planet-disk interactions |
| Profile statistics | Variance, std, dynamic range, texture (mean \|dI/dr\|) | Disk morphological complexity |
| Outer disk | Mean flux beyond 50 px | Planet-driven spiral and pressure bump extent |
| FFT spectrum | 20 low-frequency amplitudes (log1p) | Detects periodic ring spacing from resonant structures |

**Deliberately excluded:** Total flux and peak brightness — these correlate with disk mass
and distance, not planet presence, and would cluster by physical scale rather than morphology.

### Results

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Silhouette Score | **0.548** | Well-separated clusters |
| Davies-Bouldin Index | **0.481** | Compact, non-overlapping (lower is better) |
| Calinski-Harabasz | **669.2** | Strong between-cluster dispersion |

**Note on ARI:** A programmatic diagnostic confirmed that filename integers are sequential
simulation run IDs (77.8% are consecutive integers; values exceed any physically meaningful
planet count). ARI against these labels is uninformative and was omitted. All cluster
validity is assessed via the internal metrics above.

### Discovered Clusters

| Cluster | Physics Label | n | Discriminating Features |
|---------|--------------|---|------------------------|
| 0 | Multi-Ring / Planet-Rich | 102 | High ring count, extended outer emission |
| 1 | Gap-Dominated / Transitional | 43 | Highest variance and texture, single dominant gap |
| 2 | Smooth / Planet-Free | 5 | Monotonically declining profile, no substructure |

### Saved Artifacts

```
EXXA_General_Test_Outputs/
├── kmeans_k3.pkl                   # Trained K-Means model (K=3)
├── scaler.pkl                      # Fitted StandardScaler
├── umap_5d.pkl                     # Fitted 5D UMAP reducer (inference)
├── umap_2d.pkl                     # Fitted 2D UMAP reducer (visualization)
├── physics_map.pkl                 # Cluster ID → physics label mapping
├── final_results.npz               # X_scaled, X_umap_5d, X_umap_2d, labels
├── EXXA_General_Test_Report.csv    # Per-disk: Image_ID, Cluster_ID, Physics_Label, UMAP coords
├── 01_dataset_preview.png
├── 02_deprojection_example.png
├── 03_silhouette_scores.png
├── 04_umap_clusters.png
├── 05_cluster_archetypes.png
├── 06_radial_profiles.png
├── 07_feature_fingerprints.png
└── 08_example_disks.png
```

### Inference on Withheld Data

All models are saved — no retraining needed. Run only these cells in order:
**Installs → Imports → Configuration → Helper Functions → Inference cell.**

```python
import joblib

# Load saved artifacts
physics_map  = joblib.load("EXXA_General_Test_Outputs/physics_map.pkl")

# Single file
result = predict_new_disk(
    fits_path    = "path/to/disk.fits",
    model_path   = "EXXA_General_Test_Outputs/kmeans_k3.pkl",
    scaler_path  = "EXXA_General_Test_Outputs/scaler.pkl",
    umap_5d_path = "EXXA_General_Test_Outputs/umap_5d.pkl",
    physics_map  = physics_map,
)
print(result["physics_label"])   # → "Multi-Ring / Planet-Rich"

# Entire folder
df = predict_folder("path/to/fits_folder/", model_path=..., scaler_path=...,
                    umap_5d_path=..., physics_map=physics_map)
```

Identical preprocessing to training — no data leakage. Returns cluster ID,
physics label, and feature vector for each disk.

---

## Test 2 — Image-Based Test: Attention Autoencoder

**Notebook:** `EXXA_Image_Based_Test.ipynb`

### Objective

Train a convolutional autoencoder to reconstruct synthetic ALMA 1250 μm continuum
observations (600×600 px) with high-fidelity preservation of faint ring and gap
structures, and expose accessible latent space representations for downstream analysis.

### Architecture: ALMAAutoencoder

```
Input (1×600×600)
     │
 ┌───▼───────────────────────────────────────────────────────┐
 │ Enc1: Conv→BN→LeakyReLU→ResBlock→SE   32ch × 600×600     │──── skip1
 └───────────────────────┬───────────────────────────────────┘
                         │ stride-2
 ┌───────────────────────▼───────────────────────────────────┐
 │ Enc2: Conv→BN→LeakyReLU→ResBlock→SE   64ch × 300×300     │──── skip2
 └───────────────────────┬───────────────────────────────────┘
                         │ stride-2
 ┌───────────────────────▼───────────────────────────────────┐
 │ Enc3: Conv→BN→LeakyReLU→ResBlock→SE   128ch × 150×150    │
 └───────────────────────┬───────────────────────────────────┘
                         │
 ┌───────────────────────▼───────────────────────────────────┐
 │         BOTTLENECK: 128ch × 150×150  (LATENT SPACE)       │
 └───────────────────────┬───────────────────────────────────┘
                         │
 ┌───────────────────────▼───────────────────────────────────┐
 │ Up1: ConvTranspose → AttentionGate(skip2) → Dec1          │
 │                                         64ch × 300×300    │
 └───────────────────────┬───────────────────────────────────┘
                         │
 ┌───────────────────────▼───────────────────────────────────┐
 │ Up2: ConvTranspose → AttentionGate(skip1) → Dec2          │
 │                                         32ch × 600×600    │
 └───────────────────────┬───────────────────────────────────┘
                         │
                    Output (1×600×600)
```

**1,162,999 parameters · 4× spatial compression (600→150)**

### Architecture Components

**Residual Blocks** (`output = F(x) + x`): Enable stable gradient flow through
deep networks. LeakyReLU (α=0.2) preserves gradients in sparse astronomical data
where standard ReLU produces no gradients in low-activation regions.

**Squeeze-Excitation Blocks**: Learn which feature channels are important per
input — some channels encode radial ring structure, others encode azimuthal
features. Adds <1% parameters with meaningful feature quality gains.

**Attention Gates**: Weight skip connections based on relevance signalled by the
decoder, preventing raw encoder features from bypassing the bottleneck while
selectively recovering spatial detail where needed.

### Preprocessing: Handling Extreme Dynamic Range

ALMA flux values (~10⁻²¹ Jy/pixel) cause float32 numerical instability.

```python
NORM_FACTOR = 1e21

# Encoder (compress 8 decades to O(1))
x_log = torch.log1p(torch.clamp(x * NORM_FACTOR, min=0))

# Decoder (exactly reversible)
out = torch.expm1(torch.clamp(out, max=20)) / NORM_FACTOR
```

`log1p` compresses the dynamic range, preserves zero values exactly, and
is inverted without error via `expm1`.

### Loss Function — Multi-Component

```
L_total = 0.25 × L_MSE  +  0.50 × L_masked  +  0.15 × L_gradient  +  0.10 × L_multiscale
```

| Component | Weight | What it targets |
|-----------|--------|----------------|
| MSE | 25% | Overall log-space pixel accuracy |
| Masked MSE | 50% | Bright disk interior (`target_log > 0.01`) — rings/gaps are ~20% of pixels but contain all the science |
| Gradient (L1 Sobel) | 15% | Sharp edge preservation at gap boundaries and ring edges |
| Multi-scale MSE | 10% | Structural consistency at full, ½, and ¼ resolution |

The 50% masked weight reflects that ring/gap regions occupy a small fraction
of image area but are the primary scientific targets.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=5e-5, weight_decay=1e-4, betas=(0.9, 0.999)) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5, mode=min) |
| Batch size | 4 (memory-constrained at 600×600) |
| Epochs | 120 · best checkpoint at epoch 107 |
| Gradient clipping | max norm 1.0 |
| Train / val split | 90% / 10% → 135 / 15 samples |
| Augmentation | Random rotation 0–360° + horizontal flip |
| Seed | 42 |

### Results

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| MS-SSIM | **1.0000** | 0.0000 | 1.0000 | 1.0000 |
| MSE (Jy²/px²) | 4.93 × 10⁻⁴⁴ | 1.29 × 10⁻⁴³ | 1.40 × 10⁻⁴⁵ | 1.16 × 10⁻⁴² |

MS-SSIM = 1.0000 across all 150 images. Residual maps confirm non-trivial
boundary errors exist — the metric reflects structural fidelity, not identity
mapping.

### Latent Space Properties

The 128-channel 150×150 spatial latent encodes physically meaningful structure:

- **Mean activation** concentrates at the stellar position (physically correct —
  the star is the brightest, most consistently encoded feature)
- **Std deviation** shows annular ring structure with highest variance at the
  ring/gap zone (r ≈ 30–150 px), confirmed across all 150 EXXA disks

This demonstrates that 128-channel spatial representations retain sufficient
morphological detail for downstream diffusion conditioning — the direct
motivation for using this encoder in the PhysDiff proposal.

### Latent Space Access

```python
# Batch inference — returns all latents
reconstructions, latents = run_inference_on_new_data("path/to/fits/")

# Access by index
latent_i = latents[i]                     # shape: (128, 150, 150)

# Direct single-image encoding
sample = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
latent = model.get_latent_representation(sample)   # shape: (1, 128, 150, 150)

# Both methods produce identical results (verified in notebook)
```

### Saved Artifacts

```
EXXA_Image_Test_Outputs/
├── alma_autoencoder_best.pth       # Best checkpoint (epoch 107); includes training history
├── detailed_metrics.csv            # Per-image MSE and MS-SSIM for all 150 images
├── reconstruction_detailed.png     # Single sample: target / prediction / residual
├── reconstruction_grid.png         # 4-sample comparison grid
├── latent_space.png                # Mean activation, std deviation, sample encoding
└── training_curves.png             # Loss, MSE, MS-SSIM over 120 epochs (lin + log scale)
```

### Inference on Withheld Data

Run all cells up to the inference section, then change one path variable:

```python
# Default dataset
reconstructions, latents = run_inference_on_new_data()

# Withheld data — change path only
reconstructions, latents = run_inference_on_new_data("/path/to/withheld_fits/")
```

Returns `reconstructions (N, 600, 600)` and `latents (N, 128, 150, 150)`.
Per-image MSE and MS-SSIM are computed for all N images and saved to
`detailed_metrics.csv` automatically.

---

## Test 3 — Sequential Test: Transit Light Curve Classifier

**Notebook:** `EXXA_Sequential_Test.ipynb`

### Objective

Train a binary classifier to determine whether a given transit light curve shows
the presence of a planet. A physically realistic synthetic dataset is generated
using PyTransit with domain randomization matching real Kepler/TESS observation
conditions, and a 1D ResNet + Squeeze-Excitation attention classifier is trained
following Shallue & Vanderburg (2018).

### Pipeline Overview
```
Synthetic Data Generation → 70/15/15 Split → 1D ResNet + SE Attention → Training → Evaluation → Inference
  PyTransit + domain rand     stratified        1,003,601 params          OneCycleLR   ROC/AUC      withheld .npz
```

### Dataset — 40,000 Synthetic Transit Curves

| Parameter | Value |
|-----------|-------|
| Planet curves | 20,000 |
| No-planet curves | 20,000 |
| Time points per curve | 1,000 |
| Time window | ±0.25 days centered on transit |
| Transit model | PyTransit QuadraticModel (Mandel & Agol 2002) |
| Noise levels | 7 levels σ ∈ [0.0005, 0.01] — full Kepler/TESS SNR range |

**Physical parameters randomized:** Radius ratio k ∈ [0.01, 0.20], limb darkening
coefficients, transit center t0 ∈ [−0.02, 0.02] days, semi-major axis a ∈ [3, 15]
stellar radii, inclination i ∈ [83°, 90°], orbital period p ∈ [1, 30] days.

**Noise augmentations:**
- AR(1) correlated red noise (coefficient 0.3–0.8) — instrumental systematics
- Baseline drift (50% probability) — telescope thermal/pointing drift
- Data gaps (30% probability) — Kepler momentum dumps every ~3 days
- Cosmic ray spikes (30% probability) — CCD detector hits
- 20% V-shaped eclipsing binary contamination — most common Kepler false positive

**Normalization:** `flux / median(flux) - 1.0` — standard Kepler/TESS pipeline.
Median is robust to transit dip affecting only 10–20% of points.

### Model Architecture: TransitResNet
```
Input (batch, 1, 1000)
        │
   Stem: Conv1d(1→16, k=7, stride=2) → BN → ReLU → MaxPool
        │
   Layer1: ResBlock(16→32, stride=2) → ResBlock(32→32)
        │
   Layer2: ResBlock(32→64, stride=2) → ResBlock(64→64)
        │
   Layer3: ResBlock(64→128, stride=2) → ResBlock(128→128)
        │
   Layer4: ResBlock(128→256, stride=2) → ResBlock(256→256)
        │
   SE Attention: AdaptiveAvgPool → Linear(256→16) → ReLU → Linear(16→256) → Sigmoid
        │
   AdaptiveAvgPool1d(1)
        │
   Classifier: Flatten → Dropout(0.3) → Linear(256→128) → ReLU → Dropout(0.15) → Linear(128→1)
        │
   Output: (batch,) — raw logits
```

**1,003,601 parameters** · Follows Shallue & Vanderburg (2018) with SE attention added.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | OneCycleLR (10% warmup + cosine annealing) |
| Early stopping | patience=10 epochs |
| Gradient clipping | max norm 1.0 |
| Best epoch | 18 |
| Seed | 42 |

### Results

| Metric | Value |
|--------|-------|
| Test AUC | **0.9866** |
| Test Accuracy | **94.17%** |
| Low noise AUC | 0.9998 |
| Medium noise AUC | 0.9651 |
| High noise AUC | 0.9866 |
| 3-Fold CV AUC | **0.9871 ± 0.0046** |

### Note on Real Data Generalization

The classifier achieves AUC=0.9866 on held-out synthetic data with 7 noise levels
matching Kepler/TESS SNR ranges. Direct evaluation on a small set of real Kepler
and TESS phase-folded light curves showed limited generalization, which is expected
given the synthetic-to-real domain gap and the small size of available labeled real
data (n=60). Properly phase-folded Kepler confirmed planets (n=8) were correctly
classified with P>0.75.

### Saved Artifacts
```
EXXA_Sequential_Test_Inputs/
└── exxa_sequential_synthetic_dataset_40k.npz   # Pre-generated 40k dataset

EXXA_Sequential_Test_Outputs/
├── transit_resnet_best.pth          # Verified pre-trained model checkpoint
├── inference_predictions.csv        # Per-curve predictions with probabilities
├── test_metrics.csv                 # Full evaluation metrics
├── 01_dataset_visualization.png     # Synthetic dataset samples + statistics
├── 02_training_curves.png           # Loss and accuracy curves
├── 03_evaluation.png                # ROC curve + confusion matrix + per-noise ROC
├── 04_attention_visualization.png   # Input gradient saliency maps
├── 05_cross_validation.png          # 3-fold CV AUC bar chart
└── inference_results.png            # Sample inference visualizations
```

### Inference on Withheld Data

Upload your withheld `.npz` file to `MyDrive/EXXA_Sequential_Test_Inputs/`,
then run the final cell only:
```python
WITHHELD_FILE = "your_withheld_file.npz"  # ← change this
```

**Option A — Run all cells top to bottom (recommended):**
1. Set `REGENERATE = False` in the generator cell
2. Run all cells — dataset loads, model trains and saves to Drive, all plots generate

**Option B — Run inference only (minimal cells):**
1. **Drive mount cell** — mounts Google Drive
2. **Imports cell** — loads all libraries and sets seeds
3. **Configuration cell** — defines paths, constants, device
4. **Generator cell** — defines `normalize_flux`
5. **Model architecture cell** — defines `TransitResNet`
6. **Inference pipeline cell** — defines `run_inference`
7. **Final inference cell** — set `WITHHELD_FILE` and run

**Expected input format:**
- NumPy `.npz` file with key `'X'` of shape `(N, 1000)`
- Each row is one light curve with 1000 time points
- Both raw flux (median near 1.0) and pre-normalized flux (median near 0.0) are handled automatically

---

## Reproducibility

All three notebooks run end-to-end with `seed=42`.
```python
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
```

**General Test:** All models (KMeans, UMAP, StandardScaler) serialised via
`joblib`. Withheld data can be classified by loading four `.pkl` files —
no retraining and no access to training data required.

**Image-Based Test:** Pre-trained checkpoint at `alma_autoencoder_best.pth`
includes model weights, optimizer state, epoch number, best loss, and full
training history. Inference requires only the `.pth` file.

**Sequential Test:** Pre-generated dataset and pre-trained checkpoint saved to
Google Drive. Set `REGENERATE = False` to skip dataset generation. Inference
runs on withheld `.npz` files with a single path change.

---

## Installation
```bash
# General Test
pip install astropy umap-learn opencv-python-headless joblib scikit-learn scipy

# Image-Based Test
pip install astropy pytorch-msssim torch torchvision

# Sequential Test
pip install pytransit lightkurve astropy numpy pandas scikit-learn matplotlib seaborn torch
```

All notebooks install dependencies automatically in the first cell when
running on Google Colab.

---

@misc{prajapati2026exxa,
  author = {Jay Sureshkumar Prajapati},
  title  = {GSoC 2026 EXXA Test Solutions: EXXA3 and EXXA4},
  year   = {2026},
  url    = {https://github.com/coder-jayp/exxa-test-solutions-gsoc2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
