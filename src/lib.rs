//! The compiled core of frc: an exact binomial split and Fourier ring correlation, for Python
//! without NumPy. Functions take any array that supports the buffer protocol and return
//! `memoryview`s.

use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod binomial;
mod buffers;
pub mod frc;

use buffers::{Input, new_memoryview};

#[pymodule]
#[pyo3(name = "_internal")]
fn internal_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binom_split, m)?)?;
    m.add_function(wrap_pyfunction!(one_frc, m)?)?;
    m.add_function(wrap_pyfunction!(two_frc, m)?)?;
    m.add_function(wrap_pyfunction!(ring_counts, m)?)?;
    m.add_class::<FrcPlanPy>()?;
    Ok(())
}

/// The seed of a call: the caller's, or a fresh one from the operating system.
fn call_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(rand::random)
}

/// Splits every count n of an array of non-negative integers: returns, in an int32 array of the
/// same shape, a sample k of binomial(n, 0.5) for each. A Poisson-distributed count with mean
/// lambda becomes one with mean lambda / 2, and so does n - k, independently of k.
///
/// The same counts and seed give the same samples on every CPU. Without a seed, every call
/// draws a fresh one from the operating system.
#[pyfunction]
#[pyo3(signature = (a, seed=None))]
fn binom_split<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    let input = Input::get(a, "a")?;
    let shape = input.shape().to_vec();
    let seed = call_seed(seed);
    new_memoryview::<i32>(py, &shape, |out| {
        out.copy_from_slice(&input.counts()?);
        binomial::binom_split(out, seed, binomial::CpuFeatures::Detected)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// The Fourier ring correlation curve of a square image of counts, from one image: every count
/// is split as binom_split splits it, and the curve is that of the two halves, as two_frc
/// computes it. A float64 array of length n // 2. Each call makes an FrcPlan, which costs time
/// and memory; for several images of one size, make one FrcPlan and use its one_frc.
#[pyfunction]
#[pyo3(signature = (img, seed=None))]
fn one_frc<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = Input::get(img, "img")?.square_size()?;
    FrcPlanPy::new(n)?.one_frc(py, img, seed)
}

/// The Fourier ring correlation curve of two square images of one size: for each ring r < n / 2
/// of frequencies f with r <= |f| < r + 1, Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2),
/// where F1 and F2 are the images' spectra. A float64 array of length n // 2. The images are
/// read as float32.
#[pyfunction]
fn two_frc<'py>(
    py: Python<'py>,
    img1: &Bound<'py, PyAny>,
    img2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = Input::get(img1, "img1")?.square_size()?;
    FrcPlanPy::new(n)?.two_frc(py, img1, img2)
}

/// The number of frequencies in each ring r < n / 2 of an n x n spectrum: those with
/// r <= |f| < r + 1. A uint64 array of length n // 2.
#[pyfunction]
fn ring_counts(py: Python<'_>, n: usize) -> PyResult<Bound<'_, PyAny>> {
    check_size(n)?;
    let counts = frc::ring_counts(n);
    new_memoryview::<u64>(py, &[counts.len()], |out| {
        out.copy_from_slice(&counts);
        Ok(())
    })
}

fn check_size(n: usize) -> PyResult<()> {
    if n < 2 {
        return Err(PyValueError::new_err(format!(
            "images must be at least 2 x 2, not {n} x {n}"
        )));
    }
    Ok(())
}

/// What one_frc and two_frc need for n x n images: FFT plans and an n x n spectrum, made once
/// and reused for every image of that size. One plan computes one curve at a time; threads
/// that share it take turns.
#[pyclass(name = "FrcPlan", module = "rustfrc._internal", frozen)]
struct FrcPlanPy {
    plan: Mutex<frc::FrcPlan>,
}

impl FrcPlanPy {
    /// Fails unless `input` is an image of the plan's size.
    fn check_image(&self, input: &Input) -> PyResult<()> {
        let n = self.size();
        if input.square_size()? != n {
            return Err(PyValueError::new_err(format!(
                "image of shape {:?}, but the plan is for ({n}, {n})",
                input.shape()
            )));
        }
        Ok(())
    }

    fn curve_memoryview<'py>(py: Python<'py>, curve: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
        new_memoryview::<f64>(py, &[curve.len()], |out| {
            out.copy_from_slice(&curve);
            Ok(())
        })
    }
}

#[pymethods]
impl FrcPlanPy {
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        check_size(n)?;
        Ok(Self {
            plan: Mutex::new(frc::FrcPlan::new(n)),
        })
    }

    /// The side n of the images.
    #[getter]
    fn size(&self) -> usize {
        self.plan.lock().unwrap().size()
    }

    /// The number of frequencies in each ring r < n / 2, as ring_counts(n).
    fn ring_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ring_counts(py, self.size())
    }

    /// The FRC curve of an n x n image of counts, as the function one_frc computes it.
    #[pyo3(signature = (img, seed=None))]
    fn one_frc<'py>(
        &self,
        py: Python<'py>,
        img: &Bound<'py, PyAny>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = Input::get(img, "img")?;
        self.check_image(&input)?;
        let seed = call_seed(seed);
        let curve = py.detach(|| -> PyResult<Vec<f64>> {
            let counts = input.counts()?;
            let mut plan = self.plan.lock().unwrap();
            plan.one_frc(&counts, seed)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Self::curve_memoryview(py, curve)
    }

    /// The FRC curve of two n x n images, as the function two_frc computes it.
    fn two_frc<'py>(
        &self,
        py: Python<'py>,
        img1: &Bound<'py, PyAny>,
        img2: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (a, b) = (Input::get(img1, "img1")?, Input::get(img2, "img2")?);
        self.check_image(&a)?;
        self.check_image(&b)?;
        let curve = py.detach(|| -> PyResult<Vec<f64>> {
            let (a, b) = (a.floats()?, b.floats()?);
            Ok(self.plan.lock().unwrap().two_frc(&a, &b))
        })?;
        Self::curve_memoryview(py, curve)
    }
}
