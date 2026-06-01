"""
RAW Image Simulation Pipeline

Simulates what a scene would look like under different imaging conditions:
different camera spectral sensitivities, illuminants, exposures, and sensor
characteristics. See design.md for full documentation.
"""

import os
import numpy as np
from sklearn.decomposition import PCA
import colour


# ---------------------------------------------------------------------------
# Offline precomputation
# ---------------------------------------------------------------------------

class RAWSimulator:
    """Offline-precomputed state + online per-image augmentation."""

    def __init__(self, camspec_path=None):
        if camspec_path is None:
            camspec_path = os.path.join(os.path.dirname(__file__),
                                        'camspec_database.txt')
        self.wavelengths = np.arange(400, 730, 10)  # 33 points
        self._precompute(camspec_path)

    # ---- Step O-1 ~ O-4 ---------------------------------------------------

    def _load_camspec(self, path):
        """Load camspec_database.txt (camera names interleaved with data)."""
        data_rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    vals = [float(x) for x in line.split()]
                    data_rows.append(vals)
                except ValueError:
                    pass  # camera name line
        data = np.array(data_rows)  # (84, 33)
        n_cameras = len(data) // 3
        cameras = data.reshape(n_cameras, 3, 33).transpose(0, 2, 1)  # (N, 33, 3)
        return cameras

    def _precompute(self, camspec_path):
        wl = self.wavelengths

        # O-1: load camera database
        self.cameras = self._load_camspec(camspec_path)

        # O-2: fit PCA per channel
        self.pca = {}
        coeffs_all = {}
        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            X = self.cameras[:, :, ch_idx]  # (N, 33)
            p = PCA(n_components=2)
            coeffs = p.fit_transform(X)  # (N, 2)
            self.pca[ch_name] = p
            coeffs_all[ch_name] = coeffs

        # O-3: sampling statistics (Gaussian)
        self.stats = {}
        for ch_name in ['R', 'G', 'B']:
            c = coeffs_all[ch_name]
            self.stats[ch_name] = {
                'mean': c.mean(axis=0),
                'std': c.std(axis=0),
                'min': c.min(axis=0),
                'max': c.max(axis=0),
            }

        # O-4: CIE data + reference ColorChecker responses
        # CIE 1931 2-deg CMF
        cmf_obj = colour.colorimetry.MSDS_CMFS[
            'CIE 1931 2 Degree Standard Observer']
        self.cie_cmf = np.array([cmf_obj[w] for w in wl])  # (33, 3)

        # D65
        sd_D65 = colour.SDS_ILLUMINANTS['D65']
        self.d65 = np.array([sd_D65[w] for w in wl])  # (33,)

        # ColorChecker 24-patch reflectance
        cc = colour.characterisation.SDS_COLOURCHECKERS['ColorChecker N Ohta']
        cc_list = list(cc.values())
        self.cc_reflectance = np.array(
            [[sd[w] for w in wl] for sd in cc_list]).T  # (33, 24)

        # Source camera proxy: mean of the 28 cameras in the database.
        # Using CIE CMF as proxy over-estimates blue (Z peaks at 1.77 vs
        # real camera B peaks at ~0.8), which compresses the blue channel
        # dynamic range in M_cross and makes cold illuminants look too
        # neutral. The mean camera is a much better match.
        self.source_cam = self.cameras.mean(axis=0)  # (33, 3)
        self.P_A = (self.cc_reflectance * self.d65[:, None]).T \
                    @ self.source_cam * 10.0  # (24, 3)
        self.P_A /= self.P_A[:, 1:2]  # normalize G = 1

        # Precompute neutral (D65) cross-camera matrix for each possible
        # target camera — used as reference for color temp amplification.
        # Since target camera is sampled online, we store the D65 illuminant
        # spectrum instead and compute M_d65 per-image.
        self.I_d65 = self.d65 / (self.d65.max() + 1e-8)  # normalized D65

    # ---- Online augmentation (Steps 1-11) ----------------------------------

    def augment(self, raw, rng=None,
                u_range=(-3.0, 3.0),
                m_range=(50, 400),
                tint_range=(-0.04, 0.04),
                eps_range=(-0.05, 0.05),
                delta_bl=0.02,
                s_sat_range=(0.85, 1.0),
                bit_depths=(10, 12, 14, 16),
                auto_wb=True,
                cct_amplify=1.0):
        """
        Apply the full simulation pipeline to a linear RAW image.

        Parameters
        ----------
        raw : ndarray, shape (H, W, 3), float in [0, 1]
        rng : np.random.Generator or None
        auto_wb : bool
            If True, apply diagonal white-balance normalization before the
            cross-camera transform. This compensates for the green-dominant
            channel imbalance typical of un-white-balanced RAW images.
        cct_amplify : float
            Amplification factor for color temperature shift. Values > 1
            exaggerate the difference from D65 (e.g., 1.5 makes cold
            scenes bluer and warm scenes more orange). Default 1.5.

        Returns
        -------
        out : ndarray, shape (H, W, 3), float in [0, 1]
        params : dict of sampled parameters (for inspection)
        """
        if rng is None:
            rng = np.random.default_rng()

        wl = self.wavelengths

        # Step 0 (optional): diagonal white-balance normalization
        # Estimate per-channel gain so that the mean of each channel is equal.
        # This maps the green-biased RAW into an approximately balanced space
        # where the CIE CMF proxy assumption is valid.
        if auto_wb:
            ch_means = raw.mean(axis=(0, 1)) + 1e-8       # (3,)
            wb_gain = ch_means[1] / ch_means               # normalize to G
            raw_balanced = raw * wb_gain[None, None, :]
        else:
            wb_gain = np.ones(3)
            raw_balanced = raw

        # Step 1: exposure
        u = rng.uniform(*u_range)
        alpha = 2.0 ** u

        # Step 2: color temperature (mired)
        m = rng.uniform(*m_range)
        T = 1e6 / m

        # Step 3: tint (uniform sampling)
        delta_uv = rng.uniform(*tint_range)

        # Step 4: blackbody spectrum
        I_T = self._planckian_spectrum(T, wl)

        # Step 5: tint offset
        # Tint (delta_uv) is a small chromaticity perturbation (~0.01 in
        # CIE 1960 UCS). Its effect on the broadband spectrum is negligible
        # compared to the color temperature range (2500K-10000K), so we
        # apply it as a diagonal gain on the final matrix instead of
        # modifying the illuminant spectrum.
        tint_gain = self._tint_to_diagonal(I_T, delta_uv)

        # Step 6: sample target camera
        S_B = self._sample_camera(rng)

        # Step 7: cross-camera matrix with color temp amplification
        M_cross = self._cross_camera_matrix(S_B, I_T)
        if cct_amplify != 1.0:
            M_d65 = self._cross_camera_matrix(S_B, self.I_d65)
            M_cross = M_d65 + cct_amplify * (M_cross - M_d65)

        # Step 8: assemble
        epsilon = rng.uniform(*eps_range, size=(3, 3))
        np.fill_diagonal(epsilon, 0.0)
        beta = rng.uniform(-delta_bl, delta_bl, size=3)
        # Apply tint as diagonal gain: diag(tint_gain) @ M_cross
        M_tinted = np.diag(tint_gain) @ M_cross
        M_full = alpha * (M_tinted + epsilon)

        # Step 9: apply transform in the white-balanced space
        H, W = raw_balanced.shape[:2]
        v_flat = raw_balanced.reshape(-1, 3)
        v_out = (M_full @ v_flat.T).T + beta[None, :]
        v_out = v_out.reshape(H, W, 3)

        # Step 9b: restore RAW-like green bias
        # Real RAW images always have green dominance (Bayer pattern: 2G vs 1R 1B).
        # Undo the auto_wb normalization so the output looks like RAW.
        if auto_wb:
            v_out = v_out / wb_gain[None, None, :]

        # Step 10: highlight roll-off
        S_sat = rng.uniform(*s_sat_range)
        v_out = S_sat * np.tanh(v_out / (S_sat + 1e-8))
        v_out = np.clip(v_out, 0.0, S_sat)

        # Step 11: quantization
        b = int(rng.choice(bit_depths))
        levels = 2 ** b - 1
        v_out = np.floor(v_out / S_sat * levels + 0.5) / levels

        params = dict(alpha=alpha, T=T, mired=m, delta_uv=delta_uv,
                      S_sat=S_sat, bit_depth=b, u=u,
                      wb_gain=wb_gain.tolist())
        return v_out.astype(np.float32), params

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _planckian_spectrum(T, wavelengths_nm):
        h = 6.62607e-34
        c = 2.99792e8
        k = 1.38065e-23
        lam = wavelengths_nm * 1e-9
        B = (2 * np.pi * h * c ** 2) / \
            (lam ** 5 * (np.expm1(h * c / (k * T * lam))))
        return B / B.max()

    @staticmethod
    def _tint_to_diagonal(I_T, delta_uv):
        """Convert tint offset to a per-channel diagonal gain in RGB.

        Tint controls the green-magenta axis (perpendicular to the
        Planckian locus in CIE 1960 UCS):
          - positive delta_uv → green: reduce R and B equally
          - negative delta_uv → magenta: boost R and B equally

        The gain is applied symmetrically to R and B so that the
        green-magenta shift is visually balanced.
        """
        # Strength: how much R,B change per unit delta_uv.
        # delta_uv ~ U(-0.04, 0.04).
        # At delta_uv=0.04, we want R,B to drop to ~0.40 (60% reduction).
        strength = 15.0
        r_ratio = 0.5  # R responds less than B to tint shift
        gain_b = 1.0 - delta_uv * strength
        gain_r = 1.0 - delta_uv * strength * r_ratio
        gain_b = max(gain_b, 0.1)
        gain_r = max(gain_r, 0.1)
        return np.array([gain_r, 1.0, gain_b])

    def _apply_tint(self, I_T, delta_uv):
        d_lambda = 10.0
        XYZ = (I_T[:, None] * self.cie_cmf * d_lambda).sum(axis=0)
        X, Y, Z = XYZ
        denom = X + 15 * Y + 3 * Z + 1e-8
        u0 = 4 * X / denom
        v0 = 6 * Y / denom
        v_new = v0 + delta_uv
        Y_new = 1.0
        X_new = Y_new * 9 * u0 / (6 * v_new + 1e-8)
        Z_new = Y_new * (12 - 3 * u0 - 30 * v_new) / (6 * v_new + 1e-8)
        return np.array([X_new, Y_new, Z_new])

    def _sample_camera(self, rng):
        S = np.zeros((len(self.wavelengths), 3))
        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            s = self.stats[ch_name]
            sigma = rng.normal(s['mean'], s['std'])
            sigma = np.clip(sigma, s['min'], s['max'])
            curve = self.pca[ch_name].inverse_transform(sigma[None])[0]
            curve = np.clip(curve, 0.0, None)
            curve /= (curve.max() + 1e-8)
            S[:, ch_idx] = curve
        return S

    def _cross_camera_matrix(self, S_B, I_T):
        d_lambda = 10.0
        # Target camera response to ColorChecker under illuminant I_T
        P_B = (self.cc_reflectance * I_T[:, None]).T \
              @ S_B * d_lambda  # (24, 3)
        P_B /= P_B[:, 1:2] + 1e-8  # normalize G = 1
        # Least squares: find M such that P_A @ M^T ≈ P_B
        M, _, _, _ = np.linalg.lstsq(self.P_A, P_B, rcond=None)
        return M.T


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_raw_image(path):
    """Load a demosaiced PNG as linear float32 in [0, 1], RGB only."""
    from PIL import Image
    img = np.array(Image.open(path)).astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # drop alpha
    img /= 255.0
    return img
