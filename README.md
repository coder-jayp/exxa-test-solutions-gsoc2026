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
| Colab Notebook | [Open in Colab](https://colab.research.google.com/drive/1fIppebJ2obFtbncZxd8J6cDr_O4kDLa8?usp=sharing) | [Open in Colab](https://colab.research.google.com/drive/1IwcIekgtpDyxjjDNQKbS8ecmciD_KIaP?usp=sharing) | Coming soon (EXXA4 only) |
| Google Drive Outputs | [View Outputs](https://drive.google.com/drive/folders/1ck1IyUC4WgEluBtMInOQbNYiq7OI0xFk?usp=sharing) | [View Outputs](https://drive.google.com/drive/folders/13FJm4BekkhtgUcbUh0kejgzCd3XULzLI?usp=sharing) | Coming soon (EXXA4 only) |

## Repository Structure

```
exxa-test-solutions-gsoc2026/
├── EXXA_General_Test.ipynb        # Unsupervised disk morphology clustering
├── EXXA_Image_Based_Test.ipynb    # Attention autoencoder for disk reconstruction
├── EXXA_Sequential_Test.ipynb     # EXXA4 only (coming soon)
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

## Reproducibility

Both notebooks run end-to-end with `seed=42`.

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

---

## Installation

```bash
# General Test
pip install astropy umap-learn opencv-python-headless joblib scikit-learn scipy

# Image-Based Test
pip install astropy pytorch-msssim torch torchvision
```

Both notebooks install dependencies automatically in the first cell when
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
