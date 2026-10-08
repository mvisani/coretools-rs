//! The Kolmogorov distribution and the one-sample Kolmogorov–Smirnov test (Numerical Recipes,
//! 3rd edition, sections 6.14.12 and 14.3.3).

/// Below this `z`, [`cdf`] uses its small-`z` series; from here on, the large-`z` one.
const SERIES_SWITCH: f64 = 1.18;
/// Iteration cap of the inverses; they converge to 1e-15 in far fewer steps.
const MAX_ITERATIONS: usize = 100;

/// `Q(z) / 2` from the large-`z` series `Σ (-1)^(k-1) e^(-2k²z²)` (three terms suffice for `z ≥ 1.18`).
fn half_sf_series(z: f64) -> f64 {
    let x = (-2.0 * z * z).exp();
    x - x.powi(4) + x.powi(9)
}

/// The cumulative distribution function `P(z)` of the Kolmogorov distribution; 0 for `z ≤ 0`.
pub fn cdf(z: f64) -> f64 {
    const PI_SQUARED_OVER_8: f64 = 1.233_700_550_136_169_8;
    const SQRT_16_OVER_PI: f64 = 2.256_758_334_191_025;
    if z <= 0.0 {
        return 0.0;
    }
    if z < SERIES_SWITCH {
        // √(2π)/z Σ e^(-(2k-1)²π²/(8z²)) with y = e^(-π²/(8z²))
        let t = PI_SQUARED_OVER_8 / (z * z);
        let y = (-t).exp();
        return SQRT_16_OVER_PI * t.sqrt() * (y + y.powi(9) + y.powi(25) + y.powi(49));
    }
    1.0 - 2.0 * half_sf_series(z)
}

/// The survival function `Q(z) = 1 - P(z)` of the Kolmogorov distribution; 1 for `z ≤ 0`.
pub fn sf(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    if z < SERIES_SWITCH {
        return 1.0 - cdf(z);
    }
    2.0 * half_sf_series(z)
}

/// The `z` with [`cdf`]`(z) = p`.
///
/// # Panics
///
/// If `p` is not in `[0, 1]`.
pub fn inverse_cdf(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "{p} is not a probability");
    inverse_sf(1.0 - p)
}

/// The `z` with [`sf`]`(z) = q`: 0 for `q = 1`, `inf` for `q = 0`.
///
/// # Panics
///
/// If `q` is not in `[0, 1]`.
pub fn inverse_sf(q: f64) -> f64 {
    assert!((0.0..=1.0).contains(&q), "{q} is not a probability");
    if q == 1.0 {
        return 0.0;
    }
    if q == 0.0 {
        return f64::INFINITY;
    }
    if q > 0.3 {
        // Solve y ln y = -π/8 (1-q)² / (1 + y⁴ + y¹²)² for y = e^(-π²/(4z²)) by Halley's method.
        let f = -core::f64::consts::FRAC_PI_8 * (1.0 - q).powi(2);
        let mut y = inverse_x_ln_x(f);
        for _ in 0..MAX_ITERATIONS {
            let ln_y = y.ln();
            let ff = f / (1.0 + y.powi(4) + y.powi(12)).powi(2);
            let u = (y * ln_y - ff) / (1.0 + ln_y);
            let t = u / f64::max(0.5, 1.0 - 0.5 * u / (y * (1.0 + ln_y)));
            y -= t;
            if (t / y).abs() <= 1e-15 {
                break;
            }
        }
        return core::f64::consts::FRAC_PI_2 / (-y.ln()).sqrt();
    }
    // Fixed-point iteration of q/2 = x - x⁴ + x⁹ - x¹⁶ + x²⁵ for x = e^(-2z²).
    let mut x: f64 = 0.03;
    for _ in 0..MAX_ITERATIONS {
        let previous = x;
        x = 0.5 * q + x.powi(4) - x.powi(9);
        if x > 0.06 {
            x += x.powi(16) - x.powi(25);
        }
        if ((previous - x) / x).abs() <= 1e-15 {
            break;
        }
    }
    (-0.5 * x.ln()).sqrt()
}

/// The smaller root `x ∈ (0, 1/e)` of `x ln x = y`, for `-1/e < y < 0`.
fn inverse_x_ln_x(y: f64) -> f64 {
    const INV_E: f64 = 0.367_879_441_171_442_32;
    debug_assert!(-INV_E < y && y < 0.0);
    // u = ln x, starting from the inverse Taylor series near the minimum at 1/e
    let mut u = if y < -0.2 {
        (INV_E - (2.0 * INV_E * (y + INV_E)).sqrt()).ln()
    } else {
        -10.0
    };
    let mut previous_step: f64 = 0.0;
    for _ in 0..MAX_ITERATIONS {
        let step = ((y / u).ln() - u) * (u / (1.0 + u));
        u += step;
        if step < 1e-8 && (step + previous_step).abs() < 0.01 * step.abs() {
            break;
        }
        previous_step = step;
        if (step / u).abs() <= 1e-15 {
            break;
        }
    }
    u.exp()
}

/// Outcome of [`one_sample_test`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneSampleTest {
    /// The largest distance `D` between the empirical and the reference cumulative distribution.
    pub statistic: f64,
    /// `Q((√n + 0.12 + 0.11/√n) D)`; small values reject that the data follow the reference.
    pub p_value: f64,
}

/// The one-sample Kolmogorov–Smirnov test of `data` against the cumulative distribution function
/// `reference_cdf`. Sorts `data` in place.
///
/// # Panics
///
/// If `data` is empty or contains NaN.
pub fn one_sample_test(
    data: &mut [f64],
    mut reference_cdf: impl FnMut(f64) -> f64,
) -> OneSampleTest {
    assert!(
        !data.is_empty(),
        "the Kolmogorov–Smirnov test needs at least one data point"
    );
    assert!(
        !data.iter().any(|x| x.is_nan()),
        "the Kolmogorov–Smirnov test cannot rank NaN"
    );
    data.sort_unstable_by(f64::total_cmp);

    // usize → f64 (here and for j + 1) is exact up to 2^53 values, i.e. 64 PiB of data
    let n = data.len() as f64;
    let mut statistic: f64 = 0.0;
    let mut ecdf_below = 0.0;
    for (j, &x) in data.iter().enumerate() {
        // The empirical CDF steps from `ecdf_below` to `ecdf_above` at x.
        let ecdf_above = (j + 1) as f64 / n;
        let f = reference_cdf(x);
        statistic = statistic
            .max((ecdf_below - f).abs())
            .max((ecdf_above - f).abs());
        ecdf_below = ecdf_above;
    }
    let sqrt_n = n.sqrt();
    OneSampleTest {
        statistic,
        p_value: sf((sqrt_n + 0.12 + 0.11 / sqrt_n) * statistic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_close;

    /// (z, P(z), Q(z)) from the series of the Kolmogorov distribution (mpmath, 40 digits).
    const DISTRIBUTION: [(f64, f64, f64); 6] = [
        (0.3, 9.305_801_334_566_632e-6, 0.999_990_694_198_665_4),
        (0.5, 0.036_054_756_335_124_91, 0.963_945_243_664_875_1),
        (0.8, 0.455_857_588_425_801_9, 0.544_142_411_574_198_2),
        (1.18, 0.876_546_190_570_234_3, 0.123_453_809_429_765_68),
        (1.5, 0.977_782_037_383_474_9, 0.022_217_962_616_525_13),
        (2.5, 0.999_992_546_693_655_9, 7.453_306_344_157_342e-6),
    ];

    #[test]
    fn cdf_and_sf_match_the_series() {
        // C++ evaluated both at sqrt(z) where Numerical Recipes squares z.
        for (z, p, q) in DISTRIBUTION {
            assert_close(cdf(z), p);
            assert_close(sf(z), q);
        }
        assert_eq!(cdf(0.0), 0.0);
        assert_eq!(sf(0.0), 1.0);
    }

    #[test]
    fn inverses_invert_the_distribution() {
        // (q, z with Q(z) = q), bisected with mpmath
        let cases = [
            (0.999, 0.374_219_690_278_278_4),
            (0.95, 0.519_610_379_168_622_5),
            (0.5, 0.827_573_555_189_907_7),
            (0.3, 0.973_063_375_332_372_6),
            (0.1, 1.223_847_870_217_082_4),
            (0.001, 1.949_474_603_504_375_3),
        ];
        for (q, z) in cases {
            assert_close(inverse_sf(q), z);
            assert_close(inverse_cdf(1.0 - q), z);
        }
        assert_eq!(inverse_sf(1.0), 0.0);
        assert_eq!(inverse_sf(0.0), f64::INFINITY);
    }

    #[test]
    fn one_sample_test_takes_the_largest_gap_on_either_side_of_each_step() {
        // Uniform data 0, 0.5, 0.99: the largest gap is 1/3 just after 0. C++ kept the empirical CDF of
        // the step with the largest gap so far instead of the previous step, and reported 0.99 - 1/3.
        let mut data = [0.99, 0.0, 0.5];
        let test = one_sample_test(&mut data, |x| x);
        assert_eq!(data, [0.0, 0.5, 0.99]);
        assert_close(test.statistic, 1.0 / 3.0);
        // Q((√3 + 0.12 + 0.11/√3) / 3), mpmath
        assert_close(test.p_value, 0.809_557_310_616_653_3);
    }

    #[test]
    fn one_sample_statistic_agrees_with_statrs() {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        use statrs::distribution::{ContinuousCDF, Normal};
        use statrs::stats_tests::NaNPolicy;
        use statrs::stats_tests::ks_test::{KSOneSampleAlternativeMethod, ks_onesample};

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
        let sample = rand_distr::Normal::new(0.3, 1.2).unwrap();
        let reference = Normal::standard();
        for n in [1, 2, 10, 1000] {
            let mut data: Vec<f64> = sample.sample_iter(&mut rng).take(n).collect();
            let (statistic, _) = ks_onesample(
                data.clone(),
                &reference,
                KSOneSampleAlternativeMethod::TwoSidedAsymptotic,
                NaNPolicy::Error,
            )
            .unwrap();
            let test = one_sample_test(&mut data, |x| reference.cdf(x));
            assert_close(test.statistic, statistic);
        }
    }

    #[test]
    #[should_panic(expected = "needs at least one data point")]
    fn one_sample_test_of_no_data_is_a_dev_error() {
        one_sample_test(&mut [], |x| x);
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn one_sample_test_of_nan_is_a_dev_error() {
        one_sample_test(&mut [0.1, f64::NAN], |x| x);
    }
}
