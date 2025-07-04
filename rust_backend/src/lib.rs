#[macro_use]
extern crate lazy_static;

mod factorize;
use factorize::{factorize_number, factorize_range};
use pyo3::prelude::*;

#[pymodule]
fn rust_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factorize_range, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_number, m)?)?;
    Ok(())
}
