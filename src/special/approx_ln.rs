//! A fast approximate natural logarithm of probabilities.

mod table;

use table::TABLE;

use crate::probability::{LinearProbability, LogProbability};

/// Lower end of the interpolated range.
const TABLE_FROM: f64 = 0.049;
/// 1001 bins on `[0.049, 1]` (the table's last entry covers `x = 1`); 1001 is exact in `f64`.
const INV_BIN_WIDTH: f64 = (TABLE.len() - 1) as f64 / (1.0 - TABLE_FROM);

/// `ln p`, approximated by linear interpolation in a lookup table on `[0.049, 1]` with an absolute
/// error of at most 7.5e-5 (exact at 1), and exact below 0.049.
pub fn approx_ln(p: LinearProbability) -> LogProbability {
    let x = p.get();
    if x < TABLE_FROM {
        return LogProbability::new_unchecked(x.ln());
    }
    // (x - 0.049) * INV_BIN_WIDTH is in [0, 1001]: the cast truncates to the bin as intended.
    let (intercept, slope) = TABLE[((x - TABLE_FROM) * INV_BIN_WIDTH) as usize];
    LogProbability::new_unchecked(intercept + slope * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lin(p: f64) -> LinearProbability {
        LinearProbability::new(p).unwrap()
    }

    // --- ported from C++ TLogZeroOneLookupUnitTests.cpp ----------------------

    #[test]
    fn approximation_stays_within_the_table_error() {
        let n = 1_000_000;
        let start = 1e-9;
        let step = (1.0 - start) / f64::from(n);
        for i in 0..n {
            let x = start + f64::from(i) * step;
            let error = (approx_ln(lin(x)).get() - x.ln()).abs();
            assert!(error <= 7.5e-5, "ln {x}: error {error}");
        }
        assert_eq!(approx_ln(lin(1.0)).get(), 0.0);
    }

    // --- beyond the C++ tests -----------------------------------------------

    #[test]
    fn below_the_table_the_logarithm_is_exact() {
        for x in [0.0, 1e-300, 0.01, 0.048_999_999] {
            assert_eq!(approx_ln(lin(x)).get(), x.ln());
        }
    }

    #[test]
    fn approximation_is_a_log_probability_on_the_whole_table() {
        // every x of the table, including the last one below 1
        let mut x = 0.049;
        while x < 1.0 {
            assert!(approx_ln(lin(x)).is_valid(), "{x}");
            x += 1e-6;
        }
        assert!(approx_ln(lin(1.0_f64.next_down())).is_valid());
    }
}
