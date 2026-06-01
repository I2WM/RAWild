# RAW Image Simulation Pipeline

## Overview

Given a source RAW image (captured by a single camera), this pipeline simulates what the same scene would look like under different imaging conditions: different camera spectral sensitivities, different illuminants, different exposures, and different sensor characteristics. The goal is data augmentation for detection tasks.

All operations are performed in **linear RAW space** (after black level subtraction). The output is a simulated linear RAW image in `[0, 1]`.

---

## Input

- `raw`: Source RAW image, shape `(H, W, 3)`, linear, black-level-subtracted, normalized to `[0, 1]`
- `camspec_database.txt`: 28-camera spectral sensitivity database from Jiang et al. (2013)

---

## File: `camspec_database.txt` Format

The file contains spectral sensitivity data for 28 cameras. Each camera occupies **3 consecutive rows** (R, G, B channels). Each row contains **33 values** corresponding to wavelengths 400 nm to 720 nm in 10 nm steps.

```
# Total rows: 28 cameras × 3 channels = 84 rows
# Each row: 33 float values (sensitivity at 400, 410, ..., 720 nm)
# Row order per camera: R channel, G channel, B channel
```

Loading example:
```python
import numpy as np

raw_data = np.loadtxt('camspec_database.txt')  # shape: (84, 33)
cameras = raw_data.reshape(28, 3, 33)          # shape: (28, 3, 33)
# cameras[i, 0, :] -> camera i, R channel, 33 wavelength points
# cameras[i, 1, :] -> camera i, G channel
# cameras[i, 2, :] -> camera i, B channel
# Transpose to (28, 33, 3) for convenience:
cameras = cameras.transpose(0, 2, 1)           # shape: (28, 33, 3)
```

Each row is **peak-normalized** (max = 1 per channel per camera).

---

## Offline Precomputation (run once before augmentation)

### Step O-1: Load and normalize camera database

```python
cameras = load_camspec(path)       # shape: (28, 33, 3)
wavelengths = np.arange(400, 730, 10)  # 33 points, nm
```

### Step O-2: Fit PCA on each channel separately

Use `sklearn.decomposition.PCA` with `n_components=2` (Jiang et al. show 2 PCs explain >97% variance).

```python
from sklearn.decomposition import PCA

pca = {}          # dict: channel name -> fitted PCA object
coeffs_all = {}   # dict: channel name -> (28, 2) array of PCA coefficients

for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
    X = cameras[:, :, ch_idx]          # (28, 33)
    p = PCA(n_components=2)
    coeffs = p.fit_transform(X)        # (28, 2)
    pca[ch_name] = p
    coeffs_all[ch_name] = coeffs
```

### Step O-3: Compute sampling statistics from data

```python
stats = {}
for ch_name in ['R', 'G', 'B']:
    c = coeffs_all[ch_name]           # (28, 2)
    stats[ch_name] = {
        'mean': c.mean(axis=0),       # shape (2,)
        'std':  c.std(axis=0),        # shape (2,)
        'min':  c.min(axis=0),        # shape (2,) — used for clamping
        'max':  c.max(axis=0),        # shape (2,) — used for clamping
    }
```

The 28 real cameras are not uniformly distributed across the PCA rectangle; they cluster near the center. **Gaussian sampling** `N(mean, std²)` (clamped to `[min, max]`) produces target cameras that are concentrated near typical real cameras while still covering the full range. This is preferred over uniform sampling, which over-represents extreme corners that may correspond to physically implausible spectral sensitivities.

### Step O-4: Precompute reference ColorChecker responses for source camera (CIE CMF approximation)

Since the source camera's true $S_c^A(\lambda)$ is unknown, use the **mean of the 28 cameras** in the database as a proxy for the source camera. This is preferred over CIE 1931 CMFs because the CIE Z function peaks at ~1.77 (much higher than real camera blue channels ~0.8), which compresses the blue channel dynamic range in the cross-camera matrix and makes cold illuminants appear too neutral.

```python
# Load CIE 1931 2-deg CMF at 400-720 nm, 10 nm steps (used for tint computation)
# Shape: (33, 3), columns are xbar, ybar, zbar
cie_cmf = load_cie_cmf(wavelengths)

# Load Macbeth ColorChecker 24-patch reflectance spectra
# Shape: (33, 24), each column is one patch's reflectance at 400-720 nm
cc_reflectance = load_colorchecker(wavelengths)

# D65 illuminant spectrum at 400-720 nm, shape (33,)
d65 = load_D65(wavelengths)

# Source camera proxy: mean of 28 cameras in the database
source_cam = cameras.mean(axis=0)   # (33, 3)

# Source camera response to ColorChecker under D65
# P_A shape: (24, 3)
P_A = (cc_reflectance * d65[:, None]).T @ source_cam * 10.0   # (24, 3)
P_A /= P_A[:, 1:2]   # normalize so G channel = 1

# Precompute normalized D65 spectrum for color temp amplification reference
I_d65 = d65 / (d65.max() + 1e-8)
```

---

## Online Augmentation (per image)

The following steps are applied independently for each image during training.

---

### Step 0: Auto white-balance normalization

Un-white-balanced RAW images have a strong green bias (Bayer pattern: 2G vs 1R 1B, typically G/R ≈ 2.0). The cross-camera matrix assumes approximately balanced input channels. Apply diagonal white-balance normalization before the transform, and **undo it afterward** so the output retains the RAW-like green dominance.

```python
ch_means = raw.mean(axis=(0, 1)) + 1e-8       # (3,)
wb_gain = ch_means[1] / ch_means               # normalize to G channel
raw_balanced = raw * wb_gain[None, None, :]     # now all channels ≈ equal mean
```

---

### Step 1: Sample exposure

Sample a global exposure gain $\alpha$ in log-space (stops), **independently** of all other parameters:

$$u \sim \mathcal{U}(u_{\min},\, u_{\max}), \qquad \alpha = 2^u$$

```python
u_min, u_max = -3.0, 3.0        # covers 6 stops total
u = np.random.uniform(u_min, u_max)
alpha = 2.0 ** u
```

---

### Step 2: Sample color temperature (independent of exposure)

Sample color temperature $T$ in the **mired scale** $m = 10^6 / T$. Mired is preferred because equal mired intervals correspond to approximately equal perceptual chromaticity shifts along the Planckian locus.

Sample uniformly in mired:

$$m \sim \mathcal{U}(m_{\min},\, m_{\max})$$

```python
m_min = 50    # corresponds to T = 20000 K (extreme cool)
m_max = 400   # corresponds to T = 2500 K  (tungsten/candlelight)

m = np.random.uniform(m_min, m_max)
T = 1e6 / m   # color temperature in Kelvin
```

---

### Step 3: Sample tint (deviation from Planckian locus)

Real illuminants (fluorescent, LED) deviate from the ideal Planckian locus. Model this as a perturbation $\Delta uv$ controlling the **green–magenta axis** (perpendicular to the warm–cool axis):

$$\Delta uv \sim \mathcal{U}(-0.04,\, 0.04)$$

Positive $\Delta uv$ → green shift, negative → magenta shift.

```python
delta_uv = np.random.uniform(-0.04, 0.04)
```

The tint is applied as a **diagonal gain in RGB space** rather than modifying the illuminant spectrum (which is numerically unstable via CMF pseudoinverse). The gain reduces R and B symmetrically for green tint, with R responding at half the strength of B to avoid excessive redness:

```python
def tint_to_diagonal(delta_uv):
    strength = 15.0
    r_ratio = 0.5   # R responds less than B to avoid red excess
    gain_r = max(1.0 - delta_uv * strength * r_ratio, 0.1)
    gain_b = max(1.0 - delta_uv * strength, 0.1)
    return np.array([gain_r, 1.0, gain_b])

tint_gain = tint_to_diagonal(delta_uv)   # shape: (3,)
```

---

### Step 4: Compute blackbody illuminant spectrum $I_T(\lambda)$

Apply Planck's law to compute the spectral power distribution of the sampled illuminant:

$$B(\lambda, T) = \frac{2\pi h c^2}{\lambda^5 \left(\exp\!\left(\frac{hc}{kT\lambda}\right) - 1\right)}$$

```python
def planckian_spectrum(T, wavelengths_nm):
    h = 6.62607e-34   # Planck constant, J·s
    c = 2.99792e8     # speed of light, m/s
    k = 1.38065e-23   # Boltzmann constant, J/K
    lam = wavelengths_nm * 1e-9   # convert nm to m
    B = (2 * np.pi * h * c**2) / (lam**5 * (np.expm1(h * c / (k * T * lam))))
    return B / B.max()   # normalize to peak = 1

I_T = planckian_spectrum(T, wavelengths)   # shape: (33,)
```

---

### Step 5: (Tint is applied in Step 8)

Tint was previously applied by modifying the illuminant spectrum via CMF pseudoinverse, but this is numerically unstable (the pseudoinverse of a 33×3 matrix produces near-zero or negative spectra). Instead, tint is computed as a diagonal RGB gain in Step 3 and applied to the cross-camera matrix in Step 8.

---

### Step 6: Sample target camera spectral sensitivity $S_c^B(\lambda)$

Sample PCA coefficients from a **Gaussian distribution** fitted to the 28 real cameras (Step O-3), clamped to `[min, max]` to stay within the physically valid convex hull:

```python
def sample_spectral_sensitivity(pca, stats, wavelengths):
    S = np.zeros((len(wavelengths), 3))
    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        s = stats[ch_name]
        sigma = np.random.normal(s['mean'], s['std'])   # shape (2,)
        sigma = np.clip(sigma, s['min'], s['max'])       # clamp to valid range
        curve = pca[ch_name].inverse_transform(sigma[None])[0]   # (33,)
        curve = np.clip(curve, 0.0, None)   # enforce non-negativity
        curve /= (curve.max() + 1e-8)       # normalize peak to 1
        S[:, ch_idx] = curve
    return S   # shape: (33, 3)

S_B = sample_spectral_sensitivity(pca, stats, wavelengths)
```

---

### Step 7: Compute cross-camera color transformation matrix $\mathbf{M}_{A \to B, T}$

This matrix encodes both the **change in camera spectral sensitivity** ($S_c^A \to S_c^B$) and the **change in illuminant** (D65 $\to I_T$). It is computed by finding the least-squares linear mapping between ColorChecker responses under the two conditions. Tint is applied separately in Step 8.

```python
def compute_cross_camera_matrix(P_A, S_B, I_T, cc_reflectance, d_lambda=10):
    # Target camera response to ColorChecker under illuminant I_T
    P_B = (cc_reflectance * I_T[:, None]).T @ S_B * d_lambda   # (24, 3)
    P_B /= P_B[:, 1:2] + 1e-8   # normalize G = 1

    # Least squares: find M (3x3) such that P_A @ M^T ≈ P_B
    M, _, _, _ = np.linalg.lstsq(P_A, P_B, rcond=None)
    return M.T   # shape: (3, 3)

M_cross = compute_cross_camera_matrix(P_A, S_B, I_T, cc_reflectance)
```

---

### Step 8: Sample sensor noise parameters and assemble final transform matrix

```python
# Off-diagonal cross-channel mixing (models sensor spectral crosstalk)
epsilon = np.random.uniform(-0.05, 0.05, size=(3, 3))
np.fill_diagonal(epsilon, 0.0)

# Black level offset (per-channel pedestal variation)
delta = 0.02
beta = np.random.uniform(-delta, delta, size=3)

# Apply tint as diagonal gain on cross-camera matrix
# tint_gain from Step 3: [gain_r, 1.0, gain_b]
M_tinted = np.diag(tint_gain) @ M_cross

# Final matrix: exposure × (tinted cross-camera + crosstalk)
M_full = alpha * (M_tinted + epsilon)   # shape: (3, 3)
```

---

### Step 9: Apply transform and restore RAW green bias

```python
H, W = raw_balanced.shape[:2]
v_flat = raw_balanced.reshape(-1, 3)                  # (N, 3)
v_out = (M_full @ v_flat.T).T + beta[None, :]         # (N, 3)
v_out = v_out.reshape(H, W, 3)

# Undo auto white-balance to restore RAW-like green dominance.
# Real RAW images always have G > R, G > B due to the Bayer pattern.
v_out = v_out / wb_gain[None, None, :]
```

---

### Step 10: Highlight roll-off and saturation clipping

Model sensor non-linearity near full-well capacity with a soft-clip ($\tanh$), followed by hard clipping:

$$f_\gamma(v) = S \cdot \tanh\!\left(\frac{v}{S}\right)$$

```python
S_sat = np.random.uniform(0.85, 1.0)   # random saturation point

v_rolled = S_sat * np.tanh(v_out / S_sat)
v_clipped = np.clip(v_rolled, 0.0, S_sat)
```

---

### Step 11: Quantization

Quantize to a target bit depth $b$:

$$\tilde{v} = \left\lfloor \frac{v'}{S} \cdot (2^b - 1) + 0.5 \right\rfloor / (2^b - 1)$$

```python
b = np.random.choice([10, 12, 14, 16])
v_quant = np.floor(v_clipped / S_sat * (2**b - 1) + 0.5) / (2**b - 1)
```

The final output `v_quant` is the simulated RAW image in `[0, 1]`.

---

## Complete Data Flow

```
raw ∈ [0,1]  (linear, black-level-subtracted, green-biased)
    │
    ├── Step 0:  auto_wb: raw_bal = raw × wb_gain               [normalize green bias]
    │
    ├── Step 1:  sample α = 2^u,  u ~ U(-3, 3)                  [exposure]
    ├── Step 2:  sample m ~ U(50, 400),  T = 1e6/m              [color temp, mired]
    ├── Step 3:  sample Δuv ~ U(-0.04, 0.04) → tint_gain        [tint, diagonal RGB gain]
    │
    ├── Step 4:  B(λ, T) → I_T(λ)                               [blackbody spectrum]
    ├── Step 5:  (tint applied in Step 8)
    ├── Step 6:  PCA N(μ,σ²) sample → S_c^B(λ)                  [target camera, Gaussian]
    │
    ├── Step 7:  ColorChecker bridge (P_A, P_B) → M_cross       [cross-camera+illuminant]
    ├── Step 8:  M_full = α·(diag(tint) @ M_cross + ε)          [assemble w/ tint+crosstalk]
    │
    ├── Step 9:  v' = M_full @ raw_bal + β                       [apply transform]
    │            v' = v' / wb_gain                               [restore green bias]
    ├── Step 10: v'' = S·tanh(v'/S),  clip to [0, S]            [highlight roll-off]
    └── Step 11: quantize to b-bit                               [quantization]
            │
            └── simulated RAW ∈ [0,1]  (green-biased, like real RAW)
```

---

## Required Python Libraries

```
numpy
scipy
scikit-learn
colour-science   # for CIE CMF, D65 illuminant, ColorChecker reflectance
```

Loading CIE data with `colour-science`:

```python
import colour

# CIE 1931 2-deg CMF at custom wavelengths
cmf_obj = colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
cie_cmf = colour.colorimetry.msds_to_XYZ(
    cmf_obj, wavelengths=wavelengths
)   # alternative: interpolate manually

# D65 illuminant
sd_D65 = colour.SDS_ILLUMINANTS['D65']
d65 = sd_D65.values   # interpolate to your wavelengths

# Macbeth ColorChecker reflectances
cc = colour.characterisation.SDS_COLOURCHECKERS['ColorChecker 2005']
cc_reflectance = np.stack([
    sds.values for sds in cc.values()
], axis=1)   # interpolate to your wavelengths
```

Alternatively, download CIE CMF and D65 from `cvrl.org` as CSV files.

---

## Hyperparameter Summary

| Parameter | Symbol | Value | Notes |
|---|---|---|---|
| Exposure range | $u$ | $\mathcal{U}(-3,\, 3)$ stops | 6 stop range |
| Mired range | $m$ | $\mathcal{U}(50,\, 400)$ | 2500 K – 20000 K |
| Tint | $\Delta uv$ | $\mathcal{U}(-0.04,\, 0.04)$ | uniform; green–magenta axis |
| Tint strength | — | 15.0 | R,B gain per unit $\Delta uv$ |
| Tint R ratio | — | 0.5 | R responds at half B strength to avoid red excess |
| Crosstalk | $\epsilon_{ij}$ | $\mathcal{U}(-0.05,\, 0.05)$ | off-diagonal only, scales with exposure |
| Black level offset | $\beta_c$ | $\mathcal{U}(-0.02,\, 0.02)$ | per channel |
| Saturation point | $S$ | $\mathcal{U}(0.85,\, 1.0)$ | |
| Bit depth | $b$ | $\{10, 12, 14, 16\}$ | uniform choice |
| PCA components | $k$ | 2 | per channel, per Jiang et al. 2013 |
| PCA sampling | — | $\mathcal{N}(\mu,\, \sigma^2)$, clamped to $[\min, \max]$ | Gaussian, fitted from 28 cameras |
| Source camera | — | mean of 28 cameras | better than CIE CMF for blue dynamic range |
| Auto white-balance | — | undo after transform | preserves RAW green dominance |

---

## Notes for Implementation

1. **Bounding box labels are unaffected**: all operations are pixel-wise color transforms with no spatial distortion, so detection bounding boxes remain valid without modification.

2. **Batch efficiency**: Steps O-1 to O-4 (PCA fitting, ColorChecker precomputation) should be done once at startup and cached. Only Steps 0–11 run per image.

3. **PCA coefficient sampling**: Gaussian sampling `N(mean, std²)` fitted from the 28 real cameras, clamped to `[min, max]` to stay within the physically valid convex hull (Jiang et al. 2013). This concentrates samples near typical cameras while still covering extreme cases at the tails.

4. **camspec_database.txt row order**: verify the row order (R/G/B per camera) matches the format described above by checking that each row's values are non-negative and peak near 1. If the file uses a different grouping (e.g., all R channels first), reshape accordingly.

5. **Source camera proxy**: uses the mean of 28 cameras instead of CIE 1931 CMFs. The CIE Z function (blue) peaks at ~1.77, far exceeding real camera blue channels (~0.8), which compresses blue dynamic range in the cross-camera matrix.

6. **Tint implementation**: applied as a diagonal RGB gain rather than modifying the illuminant spectrum. The CMF pseudoinverse approach (projecting XYZ gains back to per-wavelength multipliers) is numerically unstable — the 33×3 pseudoinverse produces near-zero or negative spectra even with zero tint. The diagonal gain is simple, stable, and visually correct.

7. **Auto white-balance undo**: the output must retain RAW-like green dominance (G > R, G > B) because the downstream detection model expects green-biased RAW input. Without undo, the output looks like a white-balanced image with a color cast, which is unrealistic for RAW.