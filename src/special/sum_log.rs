//! Sums of logarithms that take the logarithm of products instead of every term.

use core::f64::consts::LN_2;
use core::num::FpCategory;
use core::ops::{Add, AddAssign};

/// The sum of logarithms `Σ ln xᵢ`, computed as the logarithms of products of `N` consecutive values:
/// one `ln` per `N` values.
///
/// A bin of `N` values must neither overflow nor underflow `f64`; if the values are not known to be
/// bounded accordingly, use [`DynamicSumLog`]. With `N = 1` the result is exactly the naive sum.
///
/// ```
/// use coretools_rs::special::BinnedSumLog;
///
/// let sum: BinnedSumLog<3> = [0.5, 2.0, 4.0, 0.25].into_iter().collect();
/// assert!(sum.sum().abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BinnedSumLog<const N: usize> {
    /// Logarithms of the completed bins.
    sum: f64,
    /// Product of the current bin.
    product: f64,
    in_bin: usize,
}

impl<const N: usize> BinnedSumLog<N> {
    pub const fn new() -> Self {
        const { assert!(N > 0, "a bin holds at least one value") };
        Self {
            sum: 0.0,
            product: 1.0,
            in_bin: 0,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.in_bin == N {
            self.sum += self.product.ln();
            self.product = 1.0;
            self.in_bin = 0;
        }
        self.product *= value;
        self.in_bin += 1;
    }

    /// `Σ ln xᵢ` over the values added so far; 0 if there are none.
    pub fn sum(&self) -> f64 {
        self.sum + self.product.ln()
    }
}

impl<const N: usize> Default for BinnedSumLog<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Extend<f64> for BinnedSumLog<N> {
    fn extend<I: IntoIterator<Item = f64>>(&mut self, values: I) {
        values.into_iter().for_each(|v| self.push(v));
    }
}

impl<const N: usize> FromIterator<f64> for BinnedSumLog<N> {
    fn from_iter<I: IntoIterator<Item = f64>>(values: I) -> Self {
        let mut sum = Self::new();
        sum.extend(values);
        sum
    }
}

const MANTISSA_BITS: u64 = (1 << 52) - 1;
const EXPONENT_BIAS: i64 = 1023;
/// Bit pattern of 1.0: biased exponent 1023, mantissa 0.
const ONE_BITS: u64 = 1.0_f64.to_bits();
/// 2^1000: the product of mantissas is folded into the exponent once it reaches this.
const RENORMALIZE_AT: f64 = f64::from_bits((1023 + 1000) << 52);
/// 2^64 scales subnormal values into the normal range.
const SUBNORMAL_SCALE: f64 = f64::from_bits((1023 + 64) << 52);
const SUBNORMAL_SCALE_EXPONENT: i64 = 64;

/// Splits a positive normal `x` into its mantissa in `[1, 2)` and its binary exponent.
fn split(x: f64) -> (f64, i64) {
    debug_assert!(
        x.is_normal() && x > 0.0,
        "{x} is not a positive normal number"
    );
    let bits = x.to_bits();
    // x > 0 leaves the sign bit 0, so this is the 11-bit exponent field: lossless
    let biased = (bits >> 52) as i64;
    (
        f64::from_bits((bits & MANTISSA_BITS) | ONE_BITS),
        biased - EXPONENT_BIAS,
    )
}

/// The sum of logarithms `Σ ln xᵢ`, computed as the logarithm of the product of all values, whose
/// binary exponents are kept apart as an integer so that the product can neither overflow nor
/// underflow: one `ln` in total.
///
/// Sums of disjoint sets of values combine with `+`. Zero, infinite, negative and NaN values give
/// the same `-inf`, `inf` or NaN as the naive sum.
///
/// ```
/// use coretools_rs::special::DynamicSumLog;
///
/// let sum: DynamicSumLog = [1e-300, 1e-300, 1e300].into_iter().collect();
/// assert!((sum.sum() - 1e-300_f64.ln()).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DynamicSumLog {
    /// Product of the mantissas, in `[1, 2^1000)`, or 0, `inf` or NaN once such a value was added.
    mantissa: f64,
    /// Sum of the binary exponents.
    exponent: i64,
}

impl DynamicSumLog {
    pub const fn new() -> Self {
        Self {
            mantissa: 1.0,
            exponent: 0,
        }
    }

    pub fn push(&mut self, value: f64) {
        match value.classify() {
            FpCategory::Normal if value > 0.0 => {
                let (mantissa, exponent) = split(value);
                self.mantissa *= mantissa;
                self.exponent += exponent;
                if self.mantissa >= RENORMALIZE_AT {
                    self.renormalize();
                }
            }
            FpCategory::Subnormal if value > 0.0 => {
                self.push(value * SUBNORMAL_SCALE);
                self.exponent -= SUBNORMAL_SCALE_EXPONENT;
            }
            _ if value < 0.0 => self.mantissa = f64::NAN,
            // ±0, +inf, NaN: ln of the product follows the naive sum
            _ => self.mantissa *= value,
        }
    }

    /// `Σ ln xᵢ` over the values added so far; 0 if there are none.
    pub fn sum(&self) -> f64 {
        // An exponent beyond 2^53 needs more than 8e12 values: lossless in practice.
        self.mantissa.ln() + self.exponent as f64 * LN_2
    }

    /// Moves the exponent of the mantissa product into `exponent`, leaving the product in `[1, 2)`.
    fn renormalize(&mut self) {
        if self.mantissa.is_normal() {
            let (mantissa, exponent) = split(self.mantissa);
            self.mantissa = mantissa;
            self.exponent += exponent;
        }
    }
}

impl Default for DynamicSumLog {
    fn default() -> Self {
        Self::new()
    }
}

/// The sum over the values of both sums.
impl AddAssign for DynamicSumLog {
    fn add_assign(&mut self, mut rhs: Self) {
        // Both products below 2, so that their product cannot overflow.
        self.renormalize();
        rhs.renormalize();
        self.mantissa *= rhs.mantissa;
        self.exponent += rhs.exponent;
    }
}

/// The sum over the values of both sums.
impl Add for DynamicSumLog {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl Extend<f64> for DynamicSumLog {
    fn extend<I: IntoIterator<Item = f64>>(&mut self, values: I) {
        values.into_iter().for_each(|v| self.push(v));
    }
}

impl FromIterator<f64> for DynamicSumLog {
    fn from_iter<I: IntoIterator<Item = f64>>(values: I) -> Self {
        let mut sum = Self::new();
        sum.extend(values);
        sum
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Exp1};

    use super::*;
    use crate::test_util::assert_close;

    const NUM_VALUES: usize = 10_000;

    /// `NUM_VALUES` exponential draws (as the C++ tests) and their naive sum of logarithms.
    fn exponential_values(seed: u64) -> (Vec<f64>, f64) {
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let values: Vec<f64> = Exp1.sample_iter(rng).take(NUM_VALUES).collect();
        let naive = values.iter().map(|v| v.ln()).sum();
        (values, naive)
    }

    // --- ported from C++ TSumLogTests.cpp ------------------------------------
    // Not ported: getExponentOfDouble (a private helper; no type punning here).

    #[test]
    fn bin_width_1_is_the_naive_sum() {
        let (values, naive) = exponential_values(1);
        let mut sum_log = BinnedSumLog::<1>::new();
        for &v in &values {
            sum_log.push(v);
        }
        assert_eq!(sum_log.sum(), naive);
        assert_eq!(values.into_iter().collect::<BinnedSumLog<1>>().sum(), naive);
    }

    #[test]
    fn wider_bins_match_the_naive_sum() {
        // NUM_VALUES is not a multiple of 3 or 119: the last bin is partial.
        fn check<const N: usize>(values: &[f64], naive: f64) {
            let mut sum_log = BinnedSumLog::<N>::new();
            for &v in values {
                sum_log.push(v);
            }
            assert_close(sum_log.sum(), naive);
            assert_close(
                values.iter().copied().collect::<BinnedSumLog<N>>().sum(),
                naive,
            );
        }
        let (values, naive) = exponential_values(2);
        check::<2>(&values, naive);
        check::<3>(&values, naive);
        check::<119>(&values, naive);
        check::<1000>(&values, naive);
    }

    /// The naive sum, and the dynamic sum of all values, of the even-indexed ones plus the odd-indexed
    /// ones, and collected from an iterator.
    fn dynamic_sums(values: &[f64]) -> (f64, [f64; 3]) {
        let naive = values.iter().map(|v| v.ln()).sum();
        let mut all = DynamicSumLog::new();
        let mut even = DynamicSumLog::new();
        let mut odd = DynamicSumLog::new();
        for (i, &v) in values.iter().enumerate() {
            all.push(v);
            if i % 2 == 0 {
                even.push(v);
            } else {
                odd.push(v);
            }
        }
        let collected: DynamicSumLog = values.iter().copied().collect();
        (naive, [all.sum(), (odd + even).sum(), collected.sum()])
    }

    #[test]
    fn dynamic_bins_match_the_naive_sum_across_the_whole_exponent_range() {
        // Every normal exponent: a plain product would overflow and underflow many times over.
        // as C++: 1000 mantissas 1 + i/10⁴ per exponent
        let values: Vec<f64> = (-1022..1023)
            .flat_map(|e| (0..1000).map(move |i| (1.0 + f64::from(i) / 1e4) * 2f64.powi(e)))
            .collect();
        let (naive, sums) = dynamic_sums(&values);
        for sum in sums {
            assert_close(sum, naive);
        }
    }

    #[test]
    fn dynamic_bins_match_the_naive_sum_of_random_values() {
        let (values, _) = exponential_values(3);
        let (naive, sums) = dynamic_sums(&values);
        for sum in sums {
            assert_close(sum, naive);
        }
        // C++ TSumLogProbability: values in (0, 1]
        let probabilities: Vec<f64> = values.into_iter().filter(|&v| v <= 1.0).collect();
        let (naive, sums) = dynamic_sums(&probabilities);
        for sum in sums {
            assert_close(sum, naive);
        }
    }

    // --- beyond the C++ tests -----------------------------------------------

    #[test]
    fn empty_sum_is_zero() {
        assert_eq!(BinnedSumLog::<3>::new().sum(), 0.0);
        assert_eq!(DynamicSumLog::new().sum(), 0.0);
        assert_eq!((DynamicSumLog::new() + DynamicSumLog::new()).sum(), 0.0);
    }

    #[test]
    fn dynamic_bins_keep_the_precision_of_subnormal_values() {
        // C++ multiplied 1.3 by 3e-320 into a subnormal product, keeping about 4 significant digits.
        let values = [
            1.3,
            3e-320,
            1e-310,
            7.0,
            f64::MIN_POSITIVE / 3.0,
            5e-324,
            1e300,
        ];
        let (naive, sums) = dynamic_sums(&values);
        for sum in sums {
            assert_close(sum, naive);
        }
    }

    #[test]
    fn dynamic_bins_follow_the_naive_sum_through_zero_infinity_and_negatives() {
        let cases: [(&[f64], f64); 6] = [
            (&[2.0, 0.0, 3.0], f64::NEG_INFINITY),
            (&[2.0, -0.0], f64::NEG_INFINITY),
            (&[0.5, f64::INFINITY], f64::INFINITY),
            (&[0.0, f64::INFINITY], f64::NAN),
            (&[-2.0, -3.0], f64::NAN),
            (&[2.0, f64::NAN], f64::NAN),
        ];
        for (values, expected) in cases {
            let (_, sums) = dynamic_sums(values);
            for sum in sums {
                if expected.is_nan() {
                    assert!(sum.is_nan(), "{values:?}: {sum}");
                } else {
                    assert_eq!(sum, expected, "{values:?}");
                }
            }
        }
    }
}
