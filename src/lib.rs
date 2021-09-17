use ndarray_rand::rand_distr::Binomial;
use ndarray_rand::rand::prelude::Distribution;
use ndarray_rand::rand::thread_rng;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};


#[pymodule]
fn rustfrc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    /// binom_split(a)
    /// --
    ///
    /// Takes an image (np.ndarray with dtype i32) and splits every pixel value according to the
    /// binomial distribution (n, p) with n = pixel value and p = 0.5. Returns a single, split image.
    #[pyfn(m)]
    fn binom_split<'py>(py: Python<'py>, a: PyReadonlyArrayDyn<'py, i32>) -> PyResult<&'py PyArrayDyn<i32>> {
        let mut a = a.as_array().to_owned();
        a.par_mapv_inplace(|i| {
            let mut rng = thread_rng();
            Binomial::new(i as u64, 0.5).unwrap().sample(&mut rng) as i32
        });
        Ok(a.into_pyarray(py))
    }
    Ok(())
}


