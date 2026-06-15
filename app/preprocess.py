"""
Stain normalization (optional, config-driven).

Macenko et al. (2009) H&E stain normalization, implemented in pure NumPy so it
adds no new dependency (no Docker rebuild) and is fully deterministic via a
FIXED standard reference stain matrix — no reference image to manage.

It is applied SYMMETRICALLY to gallery and query images because it lives inside
Encoder.encode() (see app/encoders/base.py), so the index and the queries can
never be normalized inconsistently. It FAILS OPEN: any patch where the stain
estimate is unstable (background-heavy, low contrast, SVD issues) is returned
unchanged rather than crashing an index build.
"""
import numpy as np
from PIL import Image


class MacenkoNormalizer:
    # Standard Macenko reference H&E stain matrix (3x2) and max concentrations.
    _HE_REF = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]])
    _MAXC_REF = np.array([1.9705, 1.0308])

    def __init__(self, Io: int = 240, alpha: float = 1.0, beta: float = 0.15):
        self.Io = Io          # transmitted light intensity
        self.alpha = alpha    # percentile for robust min/max stain angle
        self.beta = beta      # OD threshold to drop background pixels

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            return self._normalize(img)
        except Exception:
            return img  # fail open — never break a build over one odd patch

    def _normalize(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB")).astype(np.float64)
        h, w, _ = rgb.shape
        flat = rgb.reshape(-1, 3)

        # RGB -> optical density
        OD = -np.log((flat + 1.0) / self.Io)

        # Keep only tissue pixels (drop near-transparent background)
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        if ODhat.shape[0] < 10:
            return img  # not enough tissue to estimate the stain vectors

        # Two principal directions of the OD cloud
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        V = eigvecs[:, 1:3]  # eigh returns ascending eigvals -> take the 2 largest

        # Robust extreme angles in that plane define the two stain vectors
        proj = ODhat.dot(V)
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        v_min = V.dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
        v_max = V.dot(np.array([np.cos(max_phi), np.sin(max_phi)]))

        # Order so hematoxylin (heavier red component) comes first
        if v_min[0] > v_max[0]:
            HE = np.array([v_min, v_max]).T
        else:
            HE = np.array([v_max, v_min]).T

        # Stain concentrations for every pixel, then rescale to the reference maxima
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]  # (2, N)
        maxC = np.percentile(C, 99, axis=1)
        maxC[maxC == 0] = 1e-6
        C = C * (self._MAXC_REF / maxC)[:, None]

        # Reconstruct in the reference stain space
        Inorm = self.Io * np.exp(-self._HE_REF.dot(C))
        Inorm = np.clip(Inorm, 0, 255).T.reshape(h, w, 3).astype(np.uint8)
        return Image.fromarray(Inorm)


def make_stain_normalizer(method: str):
    """Return a callable(PIL.Image) -> PIL.Image, or None when disabled."""
    if not method or method == "none":
        return None
    if method == "macenko":
        return MacenkoNormalizer()
    raise ValueError(f"Unknown stain_norm '{method}'. Use 'none' or 'macenko'.")
