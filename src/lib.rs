use ndarray_rand::rand_distr::Binomial;
use ndarray_rand::rand::prelude::{Distribution, thread_rng};
use ndarray::{Array, Dimension};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyArrayDyn};
use pyo3::prelude::{pymodule, pyfunction, wrap_pyfunction, PyModule, PyResult, Python};
use pyo3::exceptions::PyValueError;
use std::convert::TryFrom;
use std::sync::atomic::{Ordering, AtomicBool};
use std::error::Error;
use std::fmt::{Display, Formatter};


#[pymodule]
fn rustfrc(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let internal = PyModule::new(py, "_internal")?;
    internal.add_function(wrap_pyfunction!(binom_split_py, internal)?)?;
    m.add_submodule(internal)?;

    Ok(())
}

/// binom_split(a)
/// --
///
/// Takes an image (np.ndarray with dtype i32) and splits every pixel value according to the
/// binomial distribution (n, p) with n = pixel value and p = 0.5. Returns a single image.
#[pyfunction]
fn binom_split_py<'py>(py: Python<'py>, a: PyReadonlyArrayDyn<'py, i32>) -> PyResult<&'py PyArrayDyn<i32>> {
    let a = a.to_owned_array();

    binom_split(a)
             .map_err(|e| PyValueError::new_err(format!("{}", e.to_string())))
             .map(|a| a.into_pyarray(py))
}

#[derive(Debug)]
struct ToUsizeError {}

impl Display for ToUsizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Value in array a cannot be cast to u64. All array \
        values must be non-negative."))
    }
}

impl Error for ToUsizeError {}

fn binom_split<D: Dimension>(mut a: Array<i32, D>) -> Result<Array<i32, D>, ToUsizeError> {
    // AtomicBool is thread-safe, and allows for communicating an error state occurred across threads
    // We initialize it with the value false, since no error occurred
    let to_unsized_failed = AtomicBool::new(false);
    // We map all values in parallel, since they do not depend on each other

    a.par_mapv_inplace(|i| {
        // If no failure has occurred, we continue
        // We use Relaxed Ordering because the order in which stores and loads occur does not matter
        // Once it is set to true, it will stay true, when exactly that happens does not matter
        if !to_unsized_failed.load(Ordering::Relaxed) {
            // We use thread rng, which is fast
            let mut rng = thread_rng();
            // We try to convert i32 to u64 (which is required for Binomial)
            // If it fails, we indicate a failure has occurred
            // Unfortunately it is not possible to escape from the loop immediately
            let n = u64::try_from(i).unwrap_or_else(|_| {
                to_unsized_failed.store(true, Ordering::Relaxed);
                0
            });
            Binomial::new(n, 0.5).unwrap().sample(&mut rng) as i32
        }
        // We just keep the rest unchanged if a failure occurred
        else {
            i
        }
    });
    let to_unsized_failed = to_unsized_failed.into_inner();
    if to_unsized_failed {
        Err(ToUsizeError {})
    } else {
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binom_split_2d() {
        let a: ndarray::Array2<i32> = ndarray::arr2(&[[9, 9, 2, 3],
            [4, 5, 6, 7]]);
        let b = binom_split(a.clone());
        let b = b.unwrap();

        assert!(b.iter().clone().max() <= a.iter().max());
        assert!(*(b.iter().min().unwrap()) >= 0);
    }

    #[test]
    fn binom_split_negative() {
        let a: ndarray::Array2<i32> = ndarray::arr2(&[[9, -9, 2, 3],
            [4, 5, 6, 7]]);
        let b = binom_split(a);

        assert!(b.is_err());
    }

    #[test]
    fn binom_split_1_element() {
        let a: ndarray::Array1<i32> = ndarray::arr1(&[0]);
        let b = binom_split(a);
        assert_eq!(b.unwrap(), ndarray::arr1(&[0]));
    }

    #[test]
    fn binom_split_large_d() {
        let a1 = ndarray::arr3(&[ [[2, 3], [4, 3]],
            [[2, 9], [4, 5]], [[9, 7], [2, 3]] ]);
        let a2 = ndarray::stack(ndarray::Axis(0), &[a1.view(), a1.view()]).unwrap();
        let a3 = ndarray::stack(ndarray::Axis(0), &[a2.view(), a2.view()]).unwrap();
        let a4 = ndarray::stack(ndarray::Axis(0), &[a3.view(), a3.view()]).unwrap();
        let a5 = ndarray::stack(ndarray::Axis(0), &[a4.view(), a4.view()]).unwrap();
        let a6 = ndarray::stack(ndarray::Axis(0), &[a5.view(), a5.view()]).unwrap();

        let b = binom_split(a6);
        assert!(b.is_ok());
    }
}