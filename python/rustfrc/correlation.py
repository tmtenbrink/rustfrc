from typing import Optional

import numpy as _np

from .rustfrc import _internal


class FrcPlan:
    """
    Fourier ring correlation of square images of one size, as frc.one_frc computes it with
    method 1: every pixel value n is split into k ~ binomial(n, 0.5) and n - k, both halves are
    Fourier transformed, and their correlation is taken over rings of equal distance from the
    origin. Ring r holds the frequencies f with round(|f|) = r, for r < n / 2.

    Making a plan allocates its buffers and plans its FFTs; reuse it for images of the same size.
    One plan computes one curve at a time; threads that share it take turns.

    :param n: The side of the square images, in pixels.
    """

    def __init__(self, n: int):
        self._plan = _internal.FrcPlan(n)

    @property
    def size(self) -> int:
        return self._plan.size

    def ring_counts(self) -> _np.ndarray:
        """The number of frequencies in each ring r < n / 2."""
        return _np.asarray(self._plan.ring_counts(), dtype=_np.uint64)

    def one_frc(self, img: _np.ndarray, seed: Optional[int] = None) -> _np.ndarray:
        """
        The FRC curve of a square image of counts: its value for each ring r < n / 2. Values are
        cast to np.int32, as binom_split does, and must not be negative.

        :param img: An n x n image of counts.
        :param seed: A seed for the split, from 0 to 2**64 - 1. The same image and seed give the
            same curve. Without one, every call splits with a fresh seed from the operating
            system, so repeated calls are independent.
        :return: The FRC curve as an np.ndarray of float64, of length n // 2.
        """
        img = _np.asarray(img)
        if img.ndim != 2 or img.shape[0] != img.shape[1]:
            raise ValueError(f"FRC needs a square 2D image, not one of shape {img.shape}")
        return self._plan.one_frc(_np.ascontiguousarray(img, dtype=_np.int32), seed)


def one_frc(img: _np.ndarray, seed: Optional[int] = None) -> _np.ndarray:
    """
    FrcPlan(img.shape[0]).one_frc(img, seed): the FRC curve of a square image of counts. Making
    the plan costs time and memory; for several images of one size, make one FrcPlan and reuse it.
    """
    img = _np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"FRC needs a square 2D image, not one of shape {img.shape}")
    return FrcPlan(img.shape[0]).one_frc(img, seed)
