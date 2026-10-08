//! Special functions and numeric helpers for likelihood code.
//!
//! Functions statrs provides are re-exported from it; the rest are ported from C++ coretools
//! (`Math/mathFunctions.h`, `TSumLog.h`, `TAccept.h`, `TAcceptOddsRation.h`, `TLogZeroOneLookup.h`):
//!
//! | C++ | here |
//! |---|---|
//! | `gammaLog`, `TDigamma::digamma` | [`ln_gamma`], [`digamma`] (statrs) |
//! | `betaLog`, `diffGammaLog`, `TTrigamma::trigamma` | [`ln_beta`], [`diff_ln_gamma`], [`trigamma`] |
//! | `TFactorial::factorial`, `factorialLog`, `choose`, `chooseLog` | [`factorial`], [`ln_factorial`], [`binomial`], [`ln_binomial`] (statrs) |
//! | `TFallingFactorial::fallingFactorial`, `fallingFactorialLog` | [`falling_factorial`], [`ln_falling_factorial`] |
//! | `TIncompleteGamma::lower`, `upper` | [`lower_incomplete_gamma`], [`upper_incomplete_gamma`] (statrs inside) |
//! | `TIncompleteBeta::incompleteBeta`, `inverseIncompleteBeta` | [`incomplete_beta`], [`inverse_incomplete_beta`] (statrs `beta_reg`, `inv_beta_reg`) |
//! | `TKolmogorovSmirnovDistr`, `runOneSampleKolmogorovSmirnovTest` | [`kolmogorov_smirnov`] |
//! | `TBinomPValue::binomPValue` | [`binomial_p_value`] |
//! | `logit`, `expit` / `logistic` | [`logit`], [`expit`] |
//! | `TSumLog<N>`, `getSumOfLog<N>` | [`BinnedSumLog`] |
//! | `TSumLog<>`, `TSumLogProbability`, `getSumOfLog`, `getSumOfLogProbability` | [`DynamicSumLog`] |
//! | `TAccept::accept`, `TAcceptOddsRatio::accept` | [`accept_log`], [`accept_logit`] |
//! | `TLogZeroOneLookup::approxLog01` | [`approx_ln`] |
//!
//! `uPow`, `logToLog10`/`log10ToLog` and `checkForNumericOverflow_*` are covered by std (`powi`,
//! `log10`/`ln`, `checked_*`).

mod accept;
mod approx_ln;
pub mod kolmogorov_smirnov;
mod sum_log;

pub use accept::{accept_log, accept_logit};
pub use approx_ln::approx_ln;
pub use sum_log::{BinnedSumLog, DynamicSumLog};

pub use statrs::function::beta::{
    beta_reg as incomplete_beta, inv_beta_reg as inverse_incomplete_beta,
};
pub use statrs::function::factorial::{binomial, factorial, ln_binomial, ln_factorial};
pub use statrs::function::gamma::{digamma, ln_gamma};

use crate::probability::LinearProbability;

/// `ln(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// From this argument on, [`ln_gamma_stirling_diff`] uses its series instead of `ln_gamma`.
const STIRLING_DIFF_USEFUL: f64 = 10.0;

/// Stirling's approximation of `ln Γ(x)`: `½ ln(2π) + (x - ½) ln(x) - x`.
fn ln_gamma_stirling(x: f64) -> f64 {
    0.5 * LN_2PI + (x - 0.5) * x.ln() - x
}

/// `ln Γ(x)` minus [`ln_gamma_stirling`]`(x)`, accurate for large `x` (Stan's `lgamma_stirling_diff`).
fn ln_gamma_stirling_diff(x: f64) -> f64 {
    const SERIES: [f64; 6] = [
        0.083_333_333_333_333_33,
        -0.002_777_777_777_777_778,
        0.000_793_650_793_650_793_7,
        -0.000_595_238_095_238_095_3,
        0.000_841_750_841_750_841_7,
        -0.001_917_526_917_526_917_5,
    ];
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < STIRLING_DIFF_USEFUL {
        return ln_gamma(x) - ln_gamma_stirling(x);
    }
    let inv_x = x.recip();
    let inv_x2 = inv_x * inv_x;
    let mut multiplier = inv_x;
    let mut result = 0.0;
    for (n, c) in SERIES.into_iter().enumerate() {
        if n > 0 {
            multiplier *= inv_x2;
        }
        result += c * multiplier;
    }
    result
}

/// `ln Γ(a) - ln Γ(a + b)` for `a, b ≥ 0`, without the catastrophic cancellation of the direct
/// difference when `a` or `b` is large.
pub fn diff_ln_gamma(a: f64, b: f64) -> f64 {
    debug_assert!(
        a >= 0.0 && b >= 0.0,
        "diff_ln_gamma({a}, {b}) needs a, b >= 0"
    );
    if a.max(b) < STIRLING_DIFF_USEFUL {
        return ln_gamma(a) - ln_gamma(a + b);
    }
    let ab = a + b;
    if a < STIRLING_DIFF_USEFUL {
        // only ln Γ(a + b) is large
        return ln_gamma(a) - ln_gamma_stirling(ab) - ln_gamma_stirling_diff(ab);
    }
    let stirling = (a - 0.5) * (-b / ab).ln_1p() + b * (1.0 - ab.ln());
    stirling + ln_gamma_stirling_diff(a) - ln_gamma_stirling_diff(ab)
}

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b)` for `a, b ≥ 0`, computed with Stirling differences
/// (Stan's `lbeta`) so that it stays accurate when either argument is large.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    debug_assert!(a >= 0.0 && b >= 0.0, "ln_beta({a}, {b}) needs a, b >= 0");
    let (x, y) = if a < b { (a, b) } else { (b, a) };
    if x == 0.0 {
        return f64::INFINITY;
    }
    if y < STIRLING_DIFF_USEFUL {
        return ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y);
    }
    let x_over_xy = x / (x + y);
    if x < STIRLING_DIFF_USEFUL {
        // y large, x small
        let stirling_diff = ln_gamma_stirling_diff(y) - ln_gamma_stirling_diff(x + y);
        let stirling = (y - 0.5) * (-x_over_xy).ln_1p() + x * (1.0 - (x + y).ln());
        return stirling + ln_gamma(x) + stirling_diff;
    }
    let stirling_diff =
        ln_gamma_stirling_diff(x) + ln_gamma_stirling_diff(y) - ln_gamma_stirling_diff(x + y);
    let stirling = (x - 0.5) * x_over_xy.ln() + y * (-x_over_xy).ln_1p() + 0.5 * (LN_2PI - y.ln());
    stirling + stirling_diff
}

/// The trigamma function `ψ₁(x) = d² ln Γ(x) / dx²`; `+inf` at the poles `x = 0, -1, -2, …`.
pub fn trigamma(x: f64) -> f64 {
    use core::f64::consts::PI;
    // Bernoulli numbers B₂, B₄, …, B₁₄ of the asymptotic series Σ B₂ₖ / x^(2k+1).
    const BERNOULLI: [f64; 7] = [
        1.0 / 6.0,
        -1.0 / 30.0,
        1.0 / 42.0,
        -1.0 / 30.0,
        5.0 / 66.0,
        -691.0 / 2730.0,
        7.0 / 6.0,
    ];
    // From here on the first omitted term is below 1e-16 of the result.
    const ASYMPTOTIC_FROM: f64 = 10.0;

    if x <= 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        // reflection: ψ₁(1 - x) + ψ₁(x) = π² / sin²(πx)
        let s = (PI * x).sin();
        return PI * PI / (s * s) - trigamma(1.0 - x);
    }
    let mut x = x;
    let mut result = 0.0;
    while x < ASYMPTOTIC_FROM {
        // ψ₁(x) = ψ₁(x + 1) + 1 / x²
        result += (x * x).recip();
        x += 1.0;
    }
    let inv = x.recip();
    let inv2 = inv * inv;
    let tail = BERNOULLI.iter().rev().fold(0.0, |acc, b| acc * inv2 + b);
    result + inv + 0.5 * inv2 + inv * inv2 * tail
}

/// The falling factorial `x (x-1) … (x-n+1)`: 1 for `n = 0`, 0 for `n > x`.
pub fn falling_factorial(x: u64, n: u64) -> f64 {
    if n > x {
        return 0.0;
    }
    // above 2^53 the u64 → f64 conversion rounds, like the product itself
    (0..n).map(|k| (x - k) as f64).product()
}

/// `ln` of [`falling_factorial`]; `f64::MIN` (the lowest finite value, standing in for `ln 0`) for
/// `n > x`.
pub fn ln_falling_factorial(x: u64, n: u64) -> f64 {
    if n > x {
        return f64::MIN;
    }
    ln_factorial(x) - ln_factorial(x - n)
}

/// The log-odds `ln(p / (1 - p))`: `-inf` at 0, `+inf` at 1.
pub fn logit(p: LinearProbability) -> f64 {
    p.odds_ratio().ln()
}

/// The logistic function `1 / (1 + e^-x)`, the inverse of [`logit`].
pub fn expit(x: f64) -> LinearProbability {
    LinearProbability::new_unchecked((1.0 + (-x).exp()).recip())
}

/// Panics unless `a ∈ (0, inf)` and `x ≥ 0` (NaN passes and yields NaN).
fn assert_incomplete_gamma_domain(a: f64, x: f64) {
    assert!(
        !(a <= 0.0 || a == f64::INFINITY || x < 0.0),
        "the incomplete gamma function needs a in (0, inf) and x >= 0, not a = {a}, x = {x}"
    );
}

/// The regularized lower incomplete gamma function `P(a, x) = γ(a, x) / Γ(a)`; 0 at `x = 0` and 1 at
/// `x = inf`.
///
/// # Panics
///
/// If `a ≤ 0`, `a = inf` or `x < 0`.
pub fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    assert_incomplete_gamma_domain(a, x);
    match x {
        0.0 => 0.0,
        f64::INFINITY => 1.0,
        _ => statrs::function::gamma::gamma_lr(a, x),
    }
}

/// The regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`; 1 at `x = 0` and 0 at
/// `x = inf`.
///
/// # Panics
///
/// If `a ≤ 0`, `a = inf` or `x < 0`.
pub fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    assert_incomplete_gamma_domain(a, x);
    match x {
        0.0 => 1.0,
        f64::INFINITY => 0.0,
        _ => statrs::function::gamma::gamma_ur(a, x),
    }
}

/// One-sided p-value of the exact binomial test against success probability ½: `P(X ≤ successes)`
/// for `X ~ Binomial(trials, ½)`; 1 when `successes ≥ trials`.
pub fn binomial_p_value(successes: u64, trials: u64) -> f64 {
    if successes >= trials {
        return 1.0;
    }
    // P(X ≤ k) = I_½(n - k, k + 1)
    // above 2^53 the u64 → f64 conversion rounds, far below the precision of the result
    incomplete_beta((trials - successes) as f64, (successes + 1) as f64, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, assert_float_eq};

    // --- ported from C++ mathFunctionsUnitTests.cpp -------------------------
    // Not ported: uPow, logToLog10 and checkForNumericOverflow_* (std: powi, log10, checked_*);
    // gammaLog, digamma, factorial, factorialLog, choose, chooseLog and incompleteBeta /
    // inverseIncompleteBeta (re-exported from statrs, which matches every C++ test value); the
    // negative-argument factorial tests (`u64` arguments).

    #[test]
    fn ln_beta_matches_r_for_small_and_large_arguments() {
        // both small
        assert_float_eq(ln_beta(1e-10, 1e-10), 23.71899811);
        assert_float_eq(ln_beta(0.2, 0.7), 1.718554829);
        assert!(ln_beta(1.0, 1.0).abs() < 1e-14);
        assert_float_eq(ln_beta(2.0, 2.0), -1.791759469);
        // a large
        assert_float_eq(ln_beta(12.0, 2.0), -5.049856007);
        assert_float_eq(ln_beta(2e20, 2.0), -93.48969808);
        // b large
        assert_float_eq(ln_beta(2.0, 12.0), -5.049856007);
        assert_float_eq(ln_beta(2.0, 2e20), -93.48969808);
        // both large
        assert_float_eq(ln_beta(12.0, 12.0), -16.60205988);
        assert_float_eq(ln_beta(2e10, 2e10), -27725887233.0);
    }

    #[test]
    fn diff_ln_gamma_matches_r_for_small_and_large_arguments() {
        // R: differences of lbeta with one shared argument, e.g. lbeta(1e-10, 1e-10) - lbeta(1e-11, 1e-10)
        let d = |a1, a2, b| diff_ln_gamma(a1, b) - diff_ln_gamma(a2, b);
        // both small
        assert_float_eq(d(1e-10, 1e-11, 1e-10), -1.704748092);
        assert_float_eq(d(0.2, 0.21, 0.7), 0.04416376716);
        assert_float_eq(d(1.0, 1.001, 1.0), 0.0009995003331);
        assert_float_eq(d(2.0, 5.0, 2.0), 1.609437912);
        // a large
        assert_float_eq(d(12.0, 15.0, 2.0), 0.4307829161);
        assert_float_eq(d(2e20, 2e19, 2.0), -4.605170186);
        // b large
        assert_float_eq(
            diff_ln_gamma(2.0, 12.0) - diff_ln_gamma(2.0, 15.0),
            8.119696,
        );
        assert_float_eq(
            diff_ln_gamma(2.0, 2e20) - diff_ln_gamma(2.0, 2e19),
            -8.280125e+21,
        );
        // both large
        assert_float_eq(d(12.0, 15.0, 12.0), 1.966112856);
        assert_float_eq(d(2e10, 2e11, 2e10), 39294054195.0);
    }

    #[test]
    fn falling_factorial_of_n_terms() {
        let cases = [
            // (x, n, x (x-1) … (x-n+1))
            (0, 0, 1.0_f64),
            (1, 0, 1.0),
            (100, 0, 1.0),
            (0, 1, 0.0),
            (1, 1, 1.0),
            (2, 1, 2.0),
            (100, 1, 100.0),
            (0, 5, 0.0),
            (4, 5, 0.0),
            (5, 5, 120.0),
            (8, 5, 6720.0),
            (15, 5, 360_360.0),
            (0, 10, 0.0),
            (9, 10, 0.0),
            (10, 10, 3_628_800.0),
            (11, 10, 39_916_800.0),
            (20, 10, 670_442_572_800.0),
        ];
        for (x, n, expected) in cases {
            assert_eq!(falling_factorial(x, n), expected, "({x})_{n}");
            if expected == 0.0 {
                // log(0) is the lowest finite value, as in C++
                assert_eq!(ln_falling_factorial(x, n), f64::MIN, "ln ({x})_{n}");
            } else {
                assert_close(ln_falling_factorial(x, n), expected.ln());
            }
        }
    }

    #[test]
    fn incomplete_gamma_matches_zipf_r() {
        // zipfR::Rgamma(x, a, lower = TRUE / FALSE), regularized
        assert_float_eq(lower_incomplete_gamma(1.8, 0.1), 0.008867713);
        assert_float_eq(lower_incomplete_gamma(0.7, 1.92), 0.9166503);
        assert_float_eq(upper_incomplete_gamma(1.8, 0.1), 0.9911323);
        assert_float_eq(upper_incomplete_gamma(0.7, 1.92), 0.08334971);
    }

    #[test]
    fn expit_inverts_logit() {
        for p in [0.0, 0.749_837_46, 1.0] {
            let p = LinearProbability::new(p).unwrap();
            assert_float_eq(expit(logit(p)).get(), p.get());
        }
        assert_eq!(logit(LinearProbability::new(0.5).unwrap()), 0.0);
        assert_eq!(expit(0.0).get(), 0.5);
    }

    // --- beyond the C++ tests -----------------------------------------------

    #[test]
    fn trigamma_matches_mpmath() {
        // mpmath.polygamma(1, x); the C++ trigamma had the 1/(6x³) term as 1/(6x⁴) (error ~1e-3).
        assert_close(trigamma(0.001), 1_000_001.642_533_195_8);
        assert_close(trigamma(0.5), 4.934_802_200_544_679);
        assert_close(trigamma(1.0), 1.644_934_066_848_226_4);
        assert_close(trigamma(2.5), 0.490_357_756_100_234_86);
        assert_close(trigamma(6.0), 0.181_322_955_737_115_33);
        assert_close(trigamma(10.0), 0.105_166_335_681_685_75);
        assert_close(trigamma(1e3), 0.001_000_500_166_666_633_3);
        assert_close(trigamma(-0.5), 8.934_802_200_544_679);
    }

    #[test]
    fn incomplete_gamma_is_total_at_the_ends_of_x() {
        assert_eq!(lower_incomplete_gamma(1.8, 0.0), 0.0);
        assert_eq!(upper_incomplete_gamma(1.8, 0.0), 1.0);
        assert_eq!(lower_incomplete_gamma(1.8, f64::INFINITY), 1.0);
        assert_eq!(upper_incomplete_gamma(1.8, f64::INFINITY), 0.0);
    }

    #[test]
    #[should_panic(expected = "needs a in (0, inf) and x >= 0, not a = 1.8, x = -0.1")]
    fn incomplete_gamma_of_negative_x_is_a_dev_error() {
        lower_incomplete_gamma(1.8, -0.1);
    }

    #[test]
    #[should_panic(expected = "needs a in (0, inf) and x >= 0, not a = 0, x = 0")]
    fn incomplete_gamma_of_non_positive_a_is_a_dev_error() {
        upper_incomplete_gamma(0.0, 0.0);
    }

    #[test]
    fn digamma_of_large_negative_arguments_terminates() {
        // C++ looped forever below -249 (its uint8_t shift counter wrapped); mpmath reference.
        assert_close(digamma(-300.5), 5.707_110_724_639_894);
    }

    #[test]
    fn factorial_beyond_170_overflows_to_infinity() {
        assert_close(factorial(170), 7.257_415_615_307_994e306);
        assert_eq!(factorial(171), f64::INFINITY);
    }

    #[test]
    fn choosing_more_than_n_has_no_way() {
        assert_eq!(binomial(3, 5), 0.0);
        assert_eq!(ln_binomial(3, 5), f64::NEG_INFINITY);
    }

    #[test]
    fn binomial_coefficients_go_beyond_170_and_u64() {
        // C++ choose threw for n > 170 and overflowed size_t (undefined behaviour) beyond 2^64.
        assert_eq!(binomial(200, 2), 19_900.0);
        assert_close(binomial(170, 85), 9.144_841_845_131_555e49);
    }

    #[test]
    fn binomial_p_value_is_the_lower_tail_under_one_half() {
        // C++ checked R's pbinom(k, n, 0.5) within 0.1; exact values (mpmath) to 1e-9.
        assert_close(binomial_p_value(2, 10), 0.054_687_5);
        assert_close(binomial_p_value(2, 20), 0.000_201_225_280_761_718_75);
        assert_close(binomial_p_value(53, 100), 0.757_940_793_196_354_2);
        assert_close(binomial_p_value(0, 1), 0.5);
        assert_close(binomial_p_value(400, 1000), 1.364_232_078_033_009_2e-10);
        assert_close(binomial_p_value(5000, 10_000), 0.503_989_323_069_691_1);
    }

    #[test]
    fn binomial_p_value_of_k_at_least_n_is_one() {
        // C++ swapped k and n through its lookup table when k > n and k + n < 100: (10, 2) gave 0.0547.
        assert_eq!(binomial_p_value(10, 2), 1.0);
        assert_eq!(binomial_p_value(7, 7), 1.0);
        assert_eq!(binomial_p_value(0, 0), 1.0);
    }
}
