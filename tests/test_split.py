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
