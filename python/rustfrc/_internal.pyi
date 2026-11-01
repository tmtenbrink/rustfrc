from collections.abc import Buffer

def binom_split(a: Buffer, seed: int | None = None) -> memoryview:
    """Splits every count n of an array of non-negative integers: returns, in an int32 array of
    the same shape, a sample k of binomial(n, 0.5) for each.

    The same counts and seed give the same samples on every CPU. Without a seed, every call
    draws a fresh one from the operating system.
    """

def one_frc(img: Buffer, seed: int | None = None) -> memoryview:
    """The Fourier ring correlation curve of a square image of counts, from one image split by
    binom_split: a float64 array of length n // 2."""

def two_frc(img1: Buffer, img2: Buffer) -> memoryview:
    """The Fourier ring correlation curve of two square images of one size: a float64 array of
    length n // 2. The images are read as float32."""

def ring_counts(n: int) -> memoryview:
    """The number of frequencies in each ring r < n / 2 of an n x n spectrum: a uint64 array of
    length n // 2."""

class FrcPlan:
    """FFT plans and an n x n spectrum, made once and reused for every image of that size."""

    def __init__(self, n: int) -> None: ...
    @property
    def size(self) -> int:
        """The side n of the images."""
    def ring_counts(self) -> memoryview:
        """ring_counts(n)."""
    def one_frc(self, img: Buffer, seed: int | None = None) -> memoryview:
        """one_frc of an n x n image of counts."""
    def two_frc(self, img1: Buffer, img2: Buffer) -> memoryview:
        """two_frc of two n x n images."""
