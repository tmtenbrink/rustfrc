import array

import numpy as np
import pytest

import rustfrc


def pattern_image(n: int, mean: float, seed: int) -> np.ndarray:
    """Poisson counts around mean with three cosine waves, at rings 5, 19 and 51."""
    y, x = np.mgrid[0:n, 0:n]
    waves = sum(a * np.cos(2 * np.pi * (u * x + v * y) / n) for u, v, a in [(3, 5, 0.4), (17, -9, 0.2), (40, 33, 0.1)])
    return np.random.default_rng(seed).poisson(mean * (1 + waves)).astype(np.int32)


def test_pattern_rings_correlate():
    frc = np.asarray(rustfrc.one_frc(pattern_image(256, 100.0, 1), seed=3))
    assert frc.shape == (128,) and frc.dtype == np.float64
    assert np.all(frc[[5, 19, 51]] > 0.9)
    assert np.median(np.abs(frc[60:])) < 0.1


def test_same_seed_same_curve():
    img = pattern_image(64, 30.0, 2)
    plan = rustfrc.FrcPlan(64)
    assert plan.size == 64
    assert plan.one_frc(img, seed=5) == plan.one_frc(img, seed=5)
    assert plan.one_frc(img, seed=5) != plan.one_frc(img, seed=6)
    # Without a seed, every call splits anew.
    assert plan.one_frc(img) != plan.one_frc(img)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint16, np.int64, np.uint64])
def test_every_integer_type_gives_the_same_curve(dtype):
    img = pattern_image(48, 20.0, 3)
    assert rustfrc.one_frc(img.astype(dtype), seed=1) == rustfrc.one_frc(img, seed=1)


def test_strided_and_contiguous_images_give_the_same_curve():
    big = pattern_image(64, 20.0, 4)
    view = big[::2, 1::2]
    assert not view.flags.c_contiguous
    assert rustfrc.one_frc(view, seed=2) == rustfrc.one_frc(np.ascontiguousarray(view), seed=2)
    assert rustfrc.one_frc(big.T, seed=2) == rustfrc.one_frc(np.ascontiguousarray(big.T), seed=2)


def test_two_frc_of_the_halves_of_a_split_sees_the_pattern():
    img = pattern_image(64, 50.0, 5)
    half = np.asarray(rustfrc.binom_split(img, seed=9))
    curve = np.asarray(rustfrc.two_frc(half, img - half))
    assert curve.shape == (32,)
    assert np.all(curve[[5, 19]] > 0.8)


def test_two_frc_of_an_image_with_itself_is_one():
    img = pattern_image(32, 50.0, 6).astype(np.float64)
    assert np.allclose(rustfrc.two_frc(img, img), 1.0)


def test_binom_split_keeps_the_shape_and_halves_the_mean():
    a = np.full((3, 50, 70), 1000, dtype=np.int64)
    k = np.asarray(rustfrc.binom_split(a, seed=1))
    assert k.shape == a.shape and k.dtype == np.int32
    assert np.all((k >= 0) & (k <= 1000))
    assert abs(k.mean() - 500) < 1.5
    assert rustfrc.binom_split(a, seed=1) == rustfrc.binom_split(a, seed=1)
    assert np.asarray(rustfrc.binom_split(np.zeros((0, 4), dtype=np.int32))).size == 0


def test_no_numpy_needed():
    img = array.array("i", [5] * 64)
    k = rustfrc.binom_split(img, seed=3)
    assert k.format == "i" and k.shape == (64,)
    assert all(0 <= v <= 5 for v in k)
    square = memoryview(array.array("q", [7] * 256)).cast("B").cast("q", [16, 16])
    curve = rustfrc.one_frc(square, seed=4)
    assert curve.format == "d" and len(curve) == 8
    counts = rustfrc.ring_counts(16)
    assert counts.format == "Q" and counts.tolist()[:2] == [1, 8]


def test_ring_counts_cover_the_disk():
    counts = np.asarray(rustfrc.FrcPlan(16).ring_counts())
    assert counts[0] == 1 and counts[1] == 8
    assert counts.sum() == sum(1 for fy in range(-8, 8) for fx in range(-8, 8) if int((fx * fx + fy * fy) ** 0.5) < 8)
    assert rustfrc.ring_counts(16) == rustfrc.FrcPlan(16).ring_counts()


@pytest.mark.parametrize("shape", [(8, 9), (8,), (2, 2, 2), (1, 1)])
def test_non_square_images_are_refused(shape):
    with pytest.raises(ValueError):
        rustfrc.one_frc(np.ones(shape, dtype=np.int32))


def test_bad_counts_are_refused():
    plan = rustfrc.FrcPlan(16)
    with pytest.raises(ValueError, match="plan is for"):
        plan.one_frc(np.ones((8, 8), dtype=np.int32))
    img = np.ones((16, 16), dtype=np.int32)
    img[3, 4] = -1
    with pytest.raises(ValueError, match="negative"):
        plan.one_frc(img)
    with pytest.raises(ValueError, match="above"):
        rustfrc.binom_split(np.array([2**31], dtype=np.int64))
    with pytest.raises(TypeError, match="integer counts"):
        plan.one_frc(np.ones((16, 16)))
    with pytest.raises(TypeError, match="buffer protocol"):
        rustfrc.binom_split([1, 2, 3])
