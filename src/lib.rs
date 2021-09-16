use ndarray_rand::rand_distr::Binomial;
use ndarray_rand::rand::prelude::Distribution;
use ndarray_rand::rand::thread_rng;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3, PyReadonlyArray2, PyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn rustfrc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn binom_split_3d<'py>(py: Python<'py>, x: PyReadonlyArray3<'py, i32>) -> PyResult<&'py PyArray3<i32>> {
        let mut x = x.as_array().to_owned();
        x.par_mapv_inplace(|i| {
            let mut rng = thread_rng();
            Binomial::new(i as u64, 0.5).unwrap().sample(&mut rng) as i32
        });
        Ok(x.into_pyarray(py))
    }

    #[pyfn(m)]
    fn binom_split_2d<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, i32>) -> PyResult<&'py PyArray2<i32>> {
        let mut x = x.as_array().to_owned();
        x.par_mapv_inplace(|i| {
            let mut rng = thread_rng();
            Binomial::new(i as u64, 0.5).unwrap().sample(&mut rng) as i32
        });
        Ok(x.into_pyarray(py))
    }
    Ok(())
}


