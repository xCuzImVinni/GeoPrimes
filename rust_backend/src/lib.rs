use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod factorize;
use factorize::{factorize_number, factorize_range};

#[pymodule]
fn rust_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factorize_range, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_number, m)?)?;
    Ok(())
}
