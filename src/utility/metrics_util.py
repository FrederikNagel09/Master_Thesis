import numpy as np
from scipy import linalg

# =============================================================================
# FID computation
# =============================================================================


def _frechet(mu1, s1, mu2, s2) -> float:
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def _fid(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    mu1, s1 = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu2, s2 = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)
    return _frechet(mu1, s1, mu2, s2)


# =============================================================================
# Uniformity score
# =============================================================================


def _uniformity_score(dist: np.ndarray) -> float:
    """KL divergence from uniform * 1000 for readability. Lower = more uniform."""
    uniform = np.ones(10) / 10
    kl = float(np.sum(dist * np.log((dist + 1e-10) / uniform)))
    return kl * 1000
