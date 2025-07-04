use primal::Sieve;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

const MAX_SIEVE: usize = 1_000_000_000;

// Global sieve for better performance
lazy_static::lazy_static! {
    static ref SIEVE: Sieve = Sieve::new(MAX_SIEVE);
}

#[pyfunction]
pub fn factorize_range(start: u64, end: u64) -> HashMap<u64, HashMap<u64, u32>> {
    (start..=end)
        .into_par_iter()
        .map(|n| (n, factorize_number(n)))
        .collect()
}

#[pyfunction]
pub fn factorize_number(n: u64) -> HashMap<u64, u32> {
    let mut factors = HashMap::new();
    let mut remainder = n;

    for p in SIEVE.primes_from(2) {
        let prime = p as u64;
        if prime * prime > remainder {
            break;
        }

        while remainder % prime == 0 {
            *factors.entry(prime).or_insert(0) += 1;
            remainder /= prime;
        }
    }

    if remainder > 1 {
        factors.insert(remainder, 1);
    }

    factors
}
