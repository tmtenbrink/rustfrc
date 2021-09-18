from _internal import binom_split as bs
import numpy as np


def binom_split(a: np.ndarray) -> np.ndarray:
    """Takes an image (some) and splits every pixel value according to the
    binomial distribution (n, p) with n = pixel value and p = 0.5. Returns a single image.
    """
    return bs(a.astype(np.int32))
