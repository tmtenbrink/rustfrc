import numpy as _np

from .rustfrc import _internal


def binom_split(a: _np.ndarray) -> _np.ndarray:
    """
    Takes an image (numpy array) and splits every pixel (element) value according to the binomial distribution (n, p)
    with n = pixel value and p = 0.5. Returns a single image. A Poisson-distributed RV becomes a Poisson-distribution
    with lambda/2. Float arrays are casted as ints.
    :param a: Input array that can be cast to np.int32
    :return: Split array.
    """
    return _internal.binom_split_py(a.astype(_np.int32))


def sqr_abs(a: _np.ndarray) -> _np.ndarray:
    """
    Takes an array (np.ndarray with dtype complex128 or complex64) and takes the absolute value and then squares it,
    element-wise. Converting from complex64 to complex32 and then running this is generally fastest.
    :param a: Input array of np.complex128 or np.complex64.
    :return: Absolute squared array.
    """
    # complex64 is two float32 (imaginary and real)
    if a.dtype == _np.complex64:
        return _internal.sqr_abs32_py(a)
    # complex128 is two float64 (imaginary and real)
    elif a.dtype == _np.complex128:
        return _internal.sqr_abs64_py(a)
    else:
        raise ValueError("Only np.csingle and np.cdouble dtypes are supported, please cast using np.astype!")


def pois_gen(lam: float, shape: tuple) -> _np.ndarray:
    """
    Generates an array (np.ndarray with dtype float64) by sampling a Poisson distribution with parameter lambda for each
    element. Takes a lambda parameter (positive) and a shape tuple of non-negative ints. Maximum number of elements in
    the array is 2500000000 (due to memory limitations), so the product of all the shape elements must be less.
    :param shape: Tuple of ints determining the output array shape.
    :param lam: Lambda parameter for Poisson distribution.
    :return: Generated random array.
    """
    too_large = 2500000000
    if _np.prod(shape) > too_large:
        raise ValueError("Array shape too large! Maximum element number is {}".format(too_large))

    return _internal.pois_gen_py(shape, lam)
