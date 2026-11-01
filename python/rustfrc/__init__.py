"""The compiled core of frc: an exact binomial split and Fourier ring correlation.

rustfrc has no dependencies. Its functions take any array that supports the buffer protocol,
such as a NumPy array, an ``array.array`` or a ``memoryview``, and return a ``memoryview``,
which ``numpy.asarray`` wraps without copying it.
"""

from ._internal import FrcPlan, binom_split, one_frc, ring_counts, two_frc

__all__ = ["FrcPlan", "binom_split", "one_frc", "ring_counts", "two_frc"]
