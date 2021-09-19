import numpy as np
from .rustfrc import _internal


def binom_split(a: np.ndarray) -> np.ndarray:
    """Takes an image (some) and splits every pixel value according to the
    binomial distribution (n, p) with n = pixel value and p = 0.5. Returns a single image.
    """
    return _internal.binom_split(a.astype(np.int32))
