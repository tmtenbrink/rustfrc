use ndarray_rand::rand_distr::Binomial;
use ndarray_rand::rand::prelude::Distribution;
use ndarray_rand::rand::thread_rng;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyArrayDyn};
use pyo3::prelude::{pymodule, pyfunction, wrap_pyfunction, PyModule, PyResult, Python};
use pyo3::exceptions::PyValueError;
use std::convert::TryFrom;
use std::sync::atomic::{AtomicI32, Ordering};

#[pymodule]
fn rustfrc(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let internal = PyModule::new(py, "_internal")?;
    internal.add_function(wrap_pyfunction!(binom_split, internal)?)?;
    m.add_submodule(internal)?;

    Ok(())
}

/// binom_split(a)
/// --
///
/// Takes an image (np.ndarray with dtype i32) and splits every pixel value according to the
/// binomial distribution (n, p) with n = pixel value and p = 0.5. Returns a single image.
#[pyfunction]
fn binom_split<'py>(py: Python<'py>, a: PyReadonlyArrayDyn<'py, i32>) -> PyResult<&'py PyArrayDyn<i32>> {
    let mut a = a.as_array().to_owned();

    let error_i = AtomicI32::new(0);
    a.par_mapv_inplace(|i| {
        if error_i.load(Ordering::Relaxed) == 0 {
            let mut rng = thread_rng();
            let n = u64::try_from(i).unwrap_or_else(|_| {
                // Since this is a parallel function, a special AtomicI32 is necessary to communicate
                // if there is a failure.
                error_i.store(i, Ordering::Relaxed);
                0
            });
            Binomial::new(n, 0.5).unwrap().sample(&mut rng) as i32
        }
        else {
            0
        }
    });
    let error_i = error_i.into_inner();
    if error_i != 0 {
        Err(PyValueError::new_err(
            format!("{i} in array a cannot be cast to u64. All array values must be non-negative.", i=error_i)))
    } else {
        Ok(a.into_pyarray(py))
    }
}


