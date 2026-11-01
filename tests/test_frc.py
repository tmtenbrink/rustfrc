import numpy as np
import pytest

import rustfrc


def pattern_image(n: int, mean: float, seed: int) -> np.ndarray:
    """Poisson counts around mean with three cosine waves, at rings 6, 19 and 52."""
    y, x = np.mgrid[0:n, 0:n]
    waves = sum(a * np.cos(2 * np.pi * (u * x + v * y) / n) for u, v, a in [(3, 5, 0.4), (17, -9, 0.2), (40, 33, 0.1)])
    return np.random.default_rng(seed).poisson(mean * (1 + waves)).astype(np.int32)


def test_pattern_rings_correlate():
    frc = rustfrc.one_frc(pattern_image(256, 100.0, 1), seed=3)
    assert frc.shape == (128,) and frc.dtype == np.float64
    assert np.all(frc[[6, 19, 52]] > 0.9)
    assert np.median(np.abs(frc[60:])) < 0.1


def test_same_seed_same_curve():
    img = pattern_image(64, 30.0, 2)
    plan = rustfrc.FrcPlan(64)
    assert np.array_equal(plan.one_frc(img, seed=5), plan.one_frc(img, seed=5))
    assert not np.array_equal(plan.one_frc(img, seed=5), plan.one_frc(img, seed=6))
    # Without a seed, every call splits anew.
    assert not np.array_equal(plan.one_frc(img), plan.one_frc(img))


def test_floats_are_cast_like_binom_split():
    img = pattern_image(32, 20.0, 3)
    assert np.array_equal(rustfrc.one_frc(img + 0.5, seed=1), rustfrc.one_frc(img, seed=1))


def test_ring_counts_cover_the_disk():
    counts = rustfrc.FrcPlan(16).ring_counts()
    assert counts[0] == 1 and counts[1] == 8
    assert counts.sum() == sum(1 for fy in range(-8, 8) for fx in range(-8, 8) if round((fx * fx + fy * fy) ** 0.5) < 8)


@pytest.mark.parametrize("shape", [(8, 9), (8,), (2, 2, 2)])
def test_non_square_images_are_refused(shape):
    with pytest.raises(ValueError):
        rustfrc.one_frc(np.ones(shape))


def test_wrong_size_and_negative_counts_are_refused():
    plan = rustfrc.FrcPlan(16)
    with pytest.raises(ValueError):
        plan.one_frc(np.ones((8, 8)))
    img = np.ones((16, 16))
    img[3, 4] = -1
    with pytest.raises(ValueError):
        plan.one_frc(img)
