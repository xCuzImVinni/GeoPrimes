use once_cell::sync::Lazy;
use primal::Sieve;
use pyo3::prelude::*; // Hinzugefügt
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

const MAX_SIEVE: usize = 1_000_000_000;

// Global sieve for better performance
static SIEVE: Lazy<Sieve> = Lazy::new(|| Sieve::new(MAX_SIEVE));

// Cache for small factorizations (up to 65535)
static SMALL_FACTOR_CACHE: Lazy<Mutex<HashMap<u64, HashMap<u64, u32>>>> = Lazy::new(|| {
    println!("Building factorization cache for 65534 numbers...");
    let start_time = std::time::Instant::now();
    let mut cache = HashMap::new();

    for n in 2..=65535 {
        cache.insert(n, factorize_small_number(n));
    }

    println!(
        "Factorization cache built in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    Mutex::new(cache)
});

fn factorize_small_number(n: u64) -> HashMap<u64, u32> {
    let mut factors = HashMap::new();
    let mut remainder = n;
    let mut p = 2;

    while p * p <= remainder {
        while remainder % p == 0 {
            *factors.entry(p).or_insert(0) += 1;
            remainder /= p;
        }
        p += 1;
    }

    if remainder > 1 {
        factors.insert(remainder, 1);
    }

    factors
}

#[pyfunction]
pub fn factorize_range(start: u64, end: u64) -> HashMap<u64, HashMap<u64, u32>> {
    let mut results = HashMap::new();

    // Split numbers into small (<=65535) and large
    let small_start = start.min(65536);
    let small_end = end.min(65535);

    // Process small numbers from cache
    if small_start <= small_end {
        let cache = SMALL_FACTOR_CACHE.lock().unwrap();
        for n in small_start..=small_end {
            if let Some(factors) = cache.get(&n) {
                results.insert(n, factors.clone());
            }
        }
    }

    // Process large numbers in parallel
    if end > 65535 {
        let large_start = start.max(65536);
        let large_results: HashMap<_, _> = (large_start..=end)
            .into_par_iter()
            .map(|n| (n, factorize_large_number(n)))
            .collect();

        results.extend(large_results);
    }

    results
}

fn factorize_large_number(n: u64) -> HashMap<u64, u32> {
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

#[pyfunction]
pub fn factorize_number(n: u64) -> HashMap<u64, u32> {
    if n <= 65535 {
        SMALL_FACTOR_CACHE.lock().unwrap().get(&n).unwrap().clone()
    } else {
        factorize_large_number(n)
    }
}
