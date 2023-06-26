use std::convert::TryFrom;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::marker::{Send, Sync};
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array, Array2, ArrayD, Axis, Dimension, Ix2, Slice};
use ndarray::parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use ndarray_rand::rand::prelude::{Distribution, thread_rng};
use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::rand::{Rng, RngCore, SeedableRng};
use ndarray_rand::rand_distr::{Binomial, Poisson, PoissonError};
use num_complex::{Complex, Complex32, Complex64};
use num_traits::Float;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyObject;
use rand_xoshiro::rand_core::OsRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[pymodule]
fn rustfrc(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let internal = PyModule::new(py, "_internal")?;
    internal.add_function(wrap_pyfunction!(binom_split_py, internal)?)?;
    internal.add_function(wrap_pyfunction!(sqr_abs32_py, internal)?)?;
    internal.add_function(wrap_pyfunction!(sqr_abs64_py, internal)?)?;
    internal.add_function(wrap_pyfunction!(pois_gen_py, internal)?)?;

    m.add_submodule(internal)?;

    Ok(())
}

/// Takes an array (np.ndarray with dtype i32) and splits every pixel value according to the
/// binomial distribution (n, p) with n = pixel value and p = 0.5. Returns a single array.
#[pyfunction]
#[pyo3(text_signature = "a, /")]
fn binom_split_py<'py>(py: Python<'py>, a: PyReadonlyArrayDyn<'py, i32>) -> PyResult<&'py PyArrayDyn<i32>> {
    let a = a.to_owned_array();

    binom_split(a)
             .map_err(|e| PyValueError::new_err(format!("{}", e.to_string())))
             .map(|a| a.into_pyarray(py))
}

/// Takes an array (np.ndarray with dtype complex64) and takes the absolute value and then squares
/// it, element-wise.
#[pyfunction]
#[pyo3(text_signature = "a, /")]
fn sqr_abs32_py<'py>(py: Python<'py>, a: PyReadonlyArrayDyn<'py, Complex32>) -> &'py PyArrayDyn<f32> {
    let a = a.to_owned_array();

    sqr_abs(a).into_pyarray(py)
}

/// Takes an array (np.ndarray with dtype complex128) and takes the absolute value and then squares
/// it, element-wise.
#[pyfunction]
#[pyo3(text_signature = "a, /")]
fn sqr_abs64_py<'py>(py: Python<'py>, a: PyReadonlyArrayDyn<'py, Complex64>) -> &'py PyArrayDyn<f64> {
    let a = a.to_owned_array();

    sqr_abs(a).into_pyarray(py)
}

/// Generates an array (np.ndarray with dtype float64) by sampling a Poisson distribution with
/// parameter lambda for each element. Takes a lambda parameter (positive) and a shape tuple of
/// non-negative ints.
#[pyfunction]
#[pyo3(text_signature = "a, /")]
fn pois_gen_py(py: Python, shape: PyObject, lambda: f64 ) -> PyResult<&PyArrayDyn<f64>> {
    let shape_vec: Vec<usize> = shape.extract(py)?;
    let shape = shape_vec.as_slice();

    pois_gen(shape, lambda)
        .map_err(|e| PyValueError::new_err(format!("{}", e.to_string())))
        .map(|a| a.into_pyarray(py))
}

#[derive(Debug)]
pub struct ToUsizeError {}

impl Display for ToUsizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Value in array a cannot be cast to u64. All array \
        values must be non-negative."))
    }
}

impl Error for ToUsizeError {}

#[inline]
pub fn binom_slit_2(mut a: Array2<i32>) -> Result<Array2<i32>, ToUsizeError> {
    //let mut i = a.axis_chunks_iter_mut(Axis(0), 100);
    let mut os_rng = OsRng::default();
    let seed: [u8; 32] = os_rng.gen();

    let c = a.axis_chunks_iter_mut(Axis(0), 50);
    c.into_par_iter().enumerate().for_each(|(index, mut s)| {
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        for _i in 0..index {
            rng.jump();
        }


        s.mapv_inplace(|n| {
            Binomial::new(n as u64, 0.5).unwrap().sample(&mut rng) as i32
        });
            ()
    });

    Ok(a)
}

/// Takes an ndarray (i32, dimension D) and splits every pixel value according to the
/// binomial distribution (n, p) with n = element value and p = 0.5. Returns a single array.
#[inline]
pub fn binom_split<D: Dimension>(mut a: Array<i32, D>) -> Result<Array<i32, D>, ToUsizeError> {
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

/// Takes an ndarray (dimension D and complex number of generic float F) and computes the absolute
/// value and then square for each element.
fn sqr_abs<D: Dimension, F: Float + Send + Sync>(mut a: Array<Complex<F>, D>) -> Array<F, D> {
    // We map all values in parallel, since they do not depend on each other
    // As there is no standard parallel map, we first map in place
    a.par_mapv_inplace(|i| { let c: Complex<F> = Complex::from({
        i.norm_sqr()
    }); c });
    // Then we get the real part
    a.mapv(|i| {
        i.re
    })
}

/// Generates an ndarray (dynamic dimension) by sampling a Poisson distribution with parameter
/// lambda for each element. Takes a lambda parameter (positive f64) and a shape slice.
fn pois_gen(shape: &[usize], lambda: f64) -> Result<ArrayD<f64>, PoissonError> {
    if !(lambda > 0.0) {
        return Err(PoissonError::ShapeTooSmall);
    }

    let mut a = ArrayD::<f64>::from_elem(shape, lambda);
    a.par_mapv_inplace(|l| {
        let mut rng = thread_rng();

        Poisson::new(l).unwrap().sample(&mut rng)
    });
    Ok(a)
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use super::*;

    #[test]
    fn binom_split_2d() {
        let a: ndarray::Array2<i32> = ndarray::arr2(&[[9, 9, 2, 3],
            [4, 5, 6, 7]]);
        let b = binom_split(a.clone());
        let b = b.unwrap();

        let x = 3;

        let y = 2000;

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

    #[test]
    fn sqr_abs_test() {
        let a: ndarray::Array2<Complex32> = ndarray::arr2(&[[Complex32::new(3., 4.),
            Complex32::new(3.32, 1.532)],
            [Complex32::new(-2., 5.), Complex32::new(582., -423.)]]);

        let b = sqr_abs(a);
        let a: ndarray::Array2<f32> = ndarray::arr2(&[[25., 13.369424], [29., 517653.]]);
        assert_eq!(a, b);
    }

    #[test]
    fn pois_gen_test() {
        let shape: &[usize] = &[2, 3, 1];
        let lam = 20.0;

        let a = pois_gen(shape, lam).unwrap();

        let int_a = a.mapv(|f| f as i64);

        assert!(*(int_a.iter().min().unwrap()) >= 0);
    }

    #[test]
    fn binom_2_test() {
        let mut a = Array2::zeros((1000, 1000));
        a.fill(100);

        let c = binom_slit_2(a).unwrap();

        println!("{:?}", c)
    }

    #[test]
    fn binom_1_test() {
        let mut a = Array2::zeros((1000, 1000));
        a.fill(100);

        let c = binom_split(a).unwrap();

        println!("{:?}", c)
    }
}