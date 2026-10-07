# Ecosystem-first port of C++ coretools; numerical tolerance instead of C++ parity

coretools-rs is a Rust-only rewrite of the C++ library [wegmannlab/coretools](https://bitbucket.org/wegmannlab/coretools) (reference checkout at commit `126744e` in the git-ignored `external/coretools`), not a line-by-line translation. Anything std or a mature crate already provides is replaced (thiserror for errors, indicatif for progress, tracing for logging, clap for command lines, rand/rand_distr/statrs for randomness and distributions, nalgebra for linear algebra, ndarray for multi-dimensional storage, flate2 for gzip, num-traits for numeric genericity). The public API is idiomatic Rust: ecosystem traits are implemented first, and crate-owned traits exist only where at least two types share behaviour that callers use polymorphically.

Deterministic functions must match C++ results within floating-point tolerance, not bit for bit. Random streams are not reproduced: the C++ used `std::mt19937` with implementation-defined standard-library distributions, so its streams were not portable even between C++ toolchains.

## Consequences

- No Rcpp/R interoperability and no C++ FFI. An R binding, if ever needed, is a separate crate.
- Seeded runs from the C++ tools cannot be replayed with this crate.
- Latent C++ bugs are fixed rather than reproduced; every behavioural difference from C++ is recorded in a divergence log.
- The ported C++ unit tests are the acceptance spec, except tests that only exercise functionality now delegated to std or a crate.
- The crate is licensed MPL-2.0, not the usual Rust MIT/Apache-2.0 pair: it is a translation of MPL-2.0 code by a non-author, so relicensing is not an option.
