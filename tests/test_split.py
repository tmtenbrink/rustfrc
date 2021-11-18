import rustfrc
import numpy as np
import pytest


def test_binom_split_negative():
    a = np.ones((5, 5, 5))*20
    a[3, 2, 1] = -1

    with pytest.raises(ValueError):
        rustfrc.binom_split(a)


def test_binom_split_ok():
    max_val = 15

    a = np.ones((3, 7))*max_val
    split_a = rustfrc.binom_split(a)

    assert np.amax(split_a) <= max_val
    assert np.amin(split_a) >= 0


def test_square_abs_64_ok():
    c = np.full((9, 2), 4+2.32j).astype(np.complex64)
    s = rustfrc.sqr_abs(c)

    assert s.dtype == np.float32


def test_square_abs_128_ok():
    c = np.full((9, 2), 2.34 - 3.1j).astype(np.complex128)
    s = rustfrc.sqr_abs(c)

    assert s.dtype == np.float64


def test_pois_gen_ok():
    lam = 20
    input_shape = (5, 3, 2)

    r = rustfrc.pois_gen(lam, input_shape)

    assert input_shape == r.shape
