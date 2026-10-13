//! Ranks, correlations, normalisation, order statistics, smoothing and root finding on slices.

use core::cmp::Ordering;
use core::ops::{Add, Div, Sub};

use num_traits::One;

use crate::probability::LinearProbability;

/// Orders numbers by [`f64::total_cmp`] and every NaN above them, whatever its sign bit.
fn nan_last(a: &f64, b: &f64) -> Ordering {
    a.is_nan().cmp(&b.is_nan()).then_with(|| a.total_cmp(b))
}

/// The ranks `1..=n` of `values` in increasing order, ties getting the average of their ranks (R's
/// `rank`). NaN ranks above every number, each NaN on its own.
///
/// ```
/// use coretools_rs::stats::ranks;
///
/// assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), [3.5, 1.0, 3.5, 2.0]);
/// ```
pub fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| nan_last(&values[a], &values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut below = 0;
    for ties in order.chunk_by(|&a, &b| values[a] == values[b]) {
        // the average of the ranks below + 1 ..= below + ties.len()
        let rank = below as f64 + (ties.len() + 1) as f64 / 2.0;
        for &position in ties {
            ranks[position] = rank;
        }
        below += ties.len();
    }
    ranks
}

/// The Pearson correlation of the pairs `(x[i], y[i])`; `None` without variation in `x` or `y`
/// (also for fewer than 2 pairs) and with NaN or infinite values.
///
/// # Panics
///
/// If `x` and `y` differ in length.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    assert_eq!(
        x.len(),
        y.len(),
        "correlation of {} with {} values",
        x.len(),
        y.len()
    );
    weighted_correlation(x.iter().zip(y).map(|(&x, &y)| (x, y, 1.0)))
}

/// The Pearson correlation of `(x, y, weight)`, each pair counting `weight` times, from centred sums;
/// `None` without variation in `x` or `y` and with NaN or infinite values.
pub(super) fn weighted_correlation(
    weighted_pairs: impl Iterator<Item = (f64, f64, f64)> + Clone,
) -> Option<f64> {
    // Welford's running means stay exact for constant values, which then have no variance
    let (mut weight, mut mean_x, mut mean_y) = (0.0, 0.0, 0.0);
    for (x, y, w) in weighted_pairs.clone().filter(|&(_, _, w)| w > 0.0) {
        weight += w;
        mean_x += w / weight * (x - mean_x);
        mean_y += w / weight * (y - mean_y);
    }
    let (mut var_x, mut var_y, mut covariance) = (0.0, 0.0, 0.0);
    for (x, y, w) in weighted_pairs {
        let (dx, dy) = (x - mean_x, y - mean_y);
        var_x += w * dx * dx;
        var_y += w * dy * dy;
        covariance += w * dx * dy;
    }
    // false for NaN as well
    (var_x > 0.0 && var_y > 0.0 && var_x.is_finite() && var_y.is_finite())
        .then(|| covariance / (var_x.sqrt() * var_y.sqrt()))
}

/// The Spearman correlation: the [Pearson correlation](pearson_correlation) of the [`ranks`].
///
/// # Panics
///
/// If `x` and `y` differ in length.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    pearson_correlation(&ranks(x), &ranks(y))
}

/// Divides `values` by their sum, so that they sum to 1, and returns the sum.
pub fn normalize(values: &mut [f64]) -> f64 {
    let total: f64 = values.iter().sum();
    for value in values.iter_mut() {
        *value /= total;
    }
    total
}

/// `ln Σ exp(xᵢ)` of the logs `xᵢ`, without overflow or underflow of the exponentials; `-inf` for no
/// values, NaN with a NaN log.
///
/// ```
/// use coretools_rs::stats::log_sum_exp;
///
/// assert!((log_sum_exp(&[-1000.0, -1000.0]) - (-1000.0 + 2f64.ln())).abs() < 1e-12);
/// ```
pub fn log_sum_exp(logs: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &log in logs {
        if log.is_nan() {
            return f64::NAN;
        }
        max = max.max(log);
    }
    if max.is_infinite() {
        // all logs -inf (or none), or one +inf
        return max;
    }
    max + logs.iter().map(|log| (log - max).exp()).sum::<f64>().ln()
}

/// Replaces the logs `xᵢ` by `exp(xᵢ) / Σ exp(xⱼ)`, which sum to 1, without overflow or underflow
/// of the exponentials.
pub fn normalize_log_sum_exp(logs: &mut [f64]) {
    let total = log_sum_exp(logs);
    for log in logs.iter_mut() {
        *log = (*log - total).exp();
    }
}

/// The median of `values`, the mean of the two middle values for an even count; `None` for no
/// values. Reorders `values` in linear time; NaN orders above every number.
///
/// ```
/// use coretools_rs::stats::median;
///
/// assert_eq!(median(&mut [5.0, 2.0, 6.0, 1.0]), Some(3.5));
/// ```
pub fn median(values: &mut [f64]) -> Option<f64> {
    let len = values.len();
    if len == 0 {
        return None;
    }
    let (below, &mut upper, _) = values.select_nth_unstable_by(len / 2, nan_last);
    if len % 2 == 1 {
        return Some(upper);
    }
    // the even count is at least 2, so the half below the upper middle value is not empty
    let lower = below.iter().copied().max_by(nan_last)?;
    Some((lower + upper) / 2.0)
}

/// The nearest-rank percentile `p` of `values`: the smallest value such that at least a fraction `p`
/// of the values are less than or equal to it (no interpolation); the smallest value for `p = 0`,
/// `None` for no values. Reorders `values` in linear time; NaN orders above every number.
///
/// ```
/// use coretools_rs::probability::LinearProbability;
/// use coretools_rs::stats::percentile;
///
/// let quartile = LinearProbability::new(0.25).unwrap();
/// assert_eq!(percentile(&mut [7.0, 8.0, 6.0, 10.0, 16.0, 8.0, 20.0, 13.0, 3.0, 15.0], quartile), Some(7.0));
/// ```
pub fn percentile(values: &mut [f64], p: LinearProbability) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    // p ∈ [0, 1], so the rank is in 0..=len
    let rank = (p.get() * values.len() as f64).ceil() as usize;
    let (_, &mut nth, _) = values.select_nth_unstable_by(rank.max(1) - 1, nan_last);
    Some(nth)
}

/// The mean of every window of `bandwidth` values on either side of each value, cut at the ends of
/// `values`.
///
/// ```
/// use coretools_rs::stats::moving_average;
///
/// assert_eq!(moving_average(&[2.0, 0.0, 1.0, 5.0], 1), [1.0, 1.0, 2.0, 3.0]);
/// ```
pub fn moving_average(values: &[f64], bandwidth: usize) -> Vec<f64> {
    if bandwidth == 0 {
        // a window of one value is that value, which the running sum would not keep exactly
        return values.to_vec();
    }
    let mut entering = values.iter();
    let mut leaving = values.iter();
    let (mut sum, mut count) = (0.0, 0_usize);
    for value in entering.by_ref().take(bandwidth) {
        sum += value;
        count += 1;
    }
    (0..values.len())
        .map(|center| {
            if let Some(value) = entering.next() {
                sum += value;
                count += 1;
            }
            if center > bandwidth {
                // the window has moved past the first value
                if let Some(value) = leaving.next() {
                    sum -= value;
                    count -= 1;
                }
            }
            sum / count as f64
        })
        .collect()
}

/// The [`median`] of every window of `bandwidth` values on either side of each value, cut at the
/// ends of `values`.
///
/// ```
/// use coretools_rs::stats::moving_median;
///
/// assert_eq!(moving_median(&[2.0, 0.0, 1.0, 5.0], 1), [1.0, 1.0, 1.0, 3.0]);
/// ```
pub fn moving_median(values: &[f64], bandwidth: usize) -> Vec<f64> {
    let mut window = Vec::with_capacity(
        bandwidth
            .saturating_mul(2)
            .saturating_add(1)
            .min(values.len()),
    );
    (0..values.len())
        .map(|center| {
            let start = center.saturating_sub(bandwidth);
            let end = center
                .saturating_add(bandwidth)
                .saturating_add(1)
                .min(values.len());
            window.clear();
            window.extend_from_slice(&values[start..end]);
            // the window holds at least the center value
            median(&mut window).unwrap_or(f64::NAN)
        })
        .collect()
}

/// A root of `f` between `a` and `b` by bisection, where `f(a)` and `f(b)` have opposite signs: the
/// end of the last bracket that lies on the side of `a`, once the bracket is at most `tolerance`
/// wide or can no longer be split; an end where `f` is 0 right away. `None` without a sign change
/// (or with NaN) between `f(a)` and `f(b)`.
///
/// For integers, a `tolerance` of 1 gives the last value on `a`'s side of the root. The distance
/// between `a` and `b` must fit `T` (for signed integers, `|b - a| <= T::MAX`).
///
/// ```
/// use coretools_rs::stats::bisect;
///
/// let root = bisect(0.0, 2.0, 1e-9, |x: f64| x * x - 2.0).unwrap();
/// assert!((root - 2f64.sqrt()).abs() < 1e-9);
/// assert_eq!(bisect(0_u32, 100, 1, |n| f64::from(n) - 41.5), Some(41));
/// ```
pub fn bisect<T, F>(mut a: T, mut b: T, tolerance: T, mut f: F) -> Option<T>
where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Div<Output = T> + One,
    F: FnMut(T) -> f64,
{
    let f_a = f(a);
    if f_a == 0.0 {
        return Some(a);
    }
    let f_b = f(b);
    if f_b == 0.0 {
        return Some(b);
    }
    if f_a.is_nan() || f_b.is_nan() || f_a.is_sign_negative() == f_b.is_sign_negative() {
        return None;
    }
    let a_is_negative = f_a.is_sign_negative();
    let two = T::one() + T::one();
    while distance(a, b) > tolerance {
        // lower end + half the distance: no overflow beyond the distance itself
        let middle = if a < b {
            a + (b - a) / two
        } else {
            b + (a - b) / two
        };
        if middle == a || middle == b {
            break;
        }
        let f_middle = f(middle);
        if f_middle == 0.0 {
            return Some(middle);
        }
        if f_middle.is_sign_negative() == a_is_negative {
            a = middle;
        } else {
            b = middle;
        }
    }
    Some(a)
}

/// `|a - b|`, also for unsigned integers.
fn distance<T: PartialOrd + Sub<Output = T>>(a: T, b: T) -> T {
    if a > b { a - b } else { b - a }
}

#[cfg(test)]
mod tests {
    use statrs::distribution::{Binomial, DiscreteCDF};

    use super::*;
    use crate::test_util::{assert_close, assert_float_eq};

    fn p(p: f64) -> LinearProbability {
        LinearProbability::new(p).unwrap()
    }

    #[track_caller]
    fn assert_all_float_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_float_eq(actual, expected);
        }
    }

    #[test]
    fn ranks_start_at_1_and_average_ties() {
        let mut values = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];
        assert_eq!(
            ranks(&values),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
        values.reverse();
        assert_eq!(
            ranks(&values),
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );

        let values = [1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0];
        assert_eq!(ranks(&values), values);

        let values = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];
        assert_eq!(
            ranks(&values),
            [1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5, 9.5, 9.5]
        );

        let values = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 0.0];
        assert_eq!(
            ranks(&values),
            [6.0, 6.0, 6.0, 3.0, 3.0, 3.0, 9.0, 9.0, 9.0, 1.0]
        );

        assert!(ranks(&[]).is_empty());
    }

    #[test]
    fn nan_orders_above_every_number() {
        // 0/0 at run time has the sign bit set on x86-64; total_cmp alone orders it below -inf
        for nan in [f64::NAN, -f64::NAN] {
            assert_eq!(
                ranks(&[1.0, nan, f64::INFINITY, f64::NEG_INFINITY]),
                [2.0, 4.0, 3.0, 1.0]
            );
            assert_eq!(median(&mut [nan, 2.0, 1.0]), Some(2.0));
            assert_eq!(median(&mut [nan, 2.0, f64::NEG_INFINITY, 1.0]), Some(1.5));
            assert!(percentile(&mut [nan, 2.0, 1.0], p(1.0)).unwrap().is_nan());
            assert_eq!(percentile(&mut [nan, 2.0, 1.0], p(0.0)), Some(1.0));
        }
    }

    #[test]
    fn pearson_correlation_of_lines_and_a_zigzag() {
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let line = |slope: f64, intercept: f64| -> Vec<f64> {
            x.iter().map(|x| intercept + slope * x).collect()
        };
        assert_float_eq(pearson_correlation(&x, &line(2.5, 0.0)).unwrap(), 1.0);
        assert_float_eq(pearson_correlation(&x, &line(-3.2, 0.0)).unwrap(), -1.0);
        assert_float_eq(pearson_correlation(&x, &line(-1.0, 9.0)).unwrap(), -1.0);
        let zigzag: Vec<f64> = x
            .iter()
            .map(|&x| if x % 2.0 == 0.0 { x } else { 0.0 })
            .collect();
        assert_float_eq(pearson_correlation(&x, &zigzag).unwrap(), 0.369_274_5);
    }

    #[test]
    fn pearson_correlation_is_exact_at_large_offsets() {
        let x: Vec<f64> = (0..10).map(|i| 1e9 + f64::from(i)).collect();
        let y: Vec<f64> = (0..10).map(|i| 1e9 - 2.0 * f64::from(i)).collect();
        assert_close(pearson_correlation(&x, &y).unwrap(), -1.0);
    }

    #[test]
    fn correlation_needs_variation_in_both() {
        assert_eq!(pearson_correlation(&[], &[]), None);
        assert_eq!(pearson_correlation(&[1.0], &[2.0]), None);
        assert_eq!(pearson_correlation(&[1.0, 2.0, 3.0], &[0.1; 3]), None);
        assert_eq!(spearman_correlation(&[1.0, 2.0, 3.0], &[0.1; 3]), None);
    }

    #[test]
    fn correlation_of_nan_or_infinite_values_is_none() {
        assert_eq!(
            pearson_correlation(&[1.0, 2.0, f64::NAN], &[1.0, 2.0, 3.0]),
            None
        );
        assert_eq!(
            pearson_correlation(&[1.0, 2.0, f64::INFINITY], &[1.0, 2.0, 3.0]),
            None
        );
    }

    #[test]
    #[should_panic(expected = "correlation of 2 with 3 values")]
    fn correlation_needs_pairs_of_values() {
        pearson_correlation(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn spearman_correlation_of_monotone_and_a_zigzag() {
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let squares: Vec<f64> = x.iter().map(|x| x * x).collect();
        let reversed: Vec<f64> = x.iter().map(|x| 9.0 - x).collect();
        assert_float_eq(spearman_correlation(&x, &squares).unwrap(), 1.0);
        assert_float_eq(spearman_correlation(&squares, &reversed).unwrap(), -1.0);
        assert_float_eq(spearman_correlation(&x, &reversed).unwrap(), -1.0);
        let zigzag: Vec<f64> = x
            .iter()
            .map(|&x| if x % 2.0 == 0.0 { x } else { 0.0 })
            .collect();
        assert_float_eq(spearman_correlation(&x, &zigzag).unwrap(), 0.273_115_5);
    }

    #[test]
    fn normalize_scales_to_sum_1() {
        let mut values = [5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(normalize(&mut values), 35.0);
        assert_float_eq(values.iter().sum(), 1.0);
        assert_eq!(values[0], 5.0 / 35.0);
    }

    #[test]
    fn log_sum_exp_of_small_and_huge_logs() {
        assert_close(
            log_sum_exp(&[1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()]),
            6.0_f64.ln(),
        );
        assert_close(log_sum_exp(&[-1e5, -1e5]), -1e5 + 2.0_f64.ln());
        // exp(710) overflows f64
        assert_close(log_sum_exp(&[710.0, 710.0]), 710.0 + 2.0_f64.ln());
        assert!(log_sum_exp(&[0.0, f64::NAN]).is_nan());
        assert!(log_sum_exp(&[f64::NEG_INFINITY, f64::NAN]).is_nan());
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY; 2]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[0.0, f64::INFINITY]), f64::INFINITY);
    }

    #[test]
    fn normalize_log_sum_exp_scales_exponentials_to_sum_1() {
        let mut values = [5.0, 6.0, 7.0, 8.0, 9.0];
        normalize_log_sum_exp(&mut values);
        assert_float_eq(values.iter().sum(), 1.0);

        let mut huge = [-10_000.0, -101_000.0, -102_000.0, -103_000.0, -104_000.0];
        normalize_log_sum_exp(&mut huge);
        assert_float_eq(huge.iter().sum(), 1.0);
        assert_eq!(huge[0], 1.0);

        normalize_log_sum_exp(&mut []);
    }

    #[test]
    fn median_of_odd_and_even_counts() {
        assert_eq!(median(&mut [5.0, 2.0, 1.0, 4.0, 3.0]), Some(3.0));
        assert_eq!(median(&mut [5.0, 2.0, 6.0, 1.0, 4.0, 3.0]), Some(3.5));
        assert_eq!(median(&mut [7.0]), Some(7.0));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn percentile_takes_the_nearest_rank() {
        let values = [20.0, 15.0, 35.0, 50.0, 40.0];
        for (prob, expected) in [
            (0.05, 15.0),
            (0.3, 20.0),
            (0.4, 20.0),
            (0.5, 35.0),
            (1.0, 50.0),
        ] {
            assert_eq!(
                percentile(&mut values.clone(), p(prob)),
                Some(expected),
                "{prob}"
            );
        }
        let values = [7.0, 8.0, 6.0, 10.0, 16.0, 8.0, 20.0, 13.0, 3.0, 15.0];
        for (prob, expected) in [(0.25, 7.0), (0.5, 8.0), (0.75, 15.0), (1.0, 20.0)] {
            assert_eq!(
                percentile(&mut values.clone(), p(prob)),
                Some(expected),
                "{prob}"
            );
        }
    }

    #[test]
    fn percentile_0_is_the_smallest_value() {
        assert_eq!(percentile(&mut [20.0, 15.0, 35.0], p(0.0)), Some(15.0));
    }

    #[test]
    fn percentile_of_nothing_is_none() {
        assert_eq!(percentile(&mut [], p(0.5)), None);
    }

    const SMOOTHED: [f64; 13] = [
        0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];

    #[test]
    fn moving_average_over_a_window_cut_at_the_ends() {
        assert_all_float_eq(
            &moving_average(&SMOOTHED, 3),
            &[
                0.75,
                1.0,
                1.166_666_666_666_67,
                1.0,
                1.285_714_285_714_29,
                1.285_714_285_714_29,
                1.142_857_142_857_14,
                1.0,
                0.714_285_714_285_714,
                0.428_571_428_571_429,
                0.5,
                0.2,
                0.25,
            ],
        );
        assert_all_float_eq(&moving_average(&[2.0, 0.0, 1.0], 5), &[1.0, 1.0, 1.0]);
        assert_eq!(moving_average(&[2.0, 0.0, 1.0], 0), [2.0, 0.0, 1.0]);
        assert!(moving_average(&[], 10).is_empty());
    }

    #[test]
    fn moving_median_over_a_window_cut_at_the_ends() {
        assert_eq!(
            moving_median(&SMOOTHED, 3),
            [
                0.5, 1.0, 1.5, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
        assert_eq!(moving_median(&[2.0, 0.0, 1.0], 5), [1.0, 1.0, 1.0]);
        assert_eq!(moving_median(&[2.0, 0.0, 1.0], 0), [2.0, 0.0, 1.0]);
        assert!(moving_median(&[], 10).is_empty());
    }

    #[test]
    fn bisect_integers_and_floats() {
        let binomial = Binomial::new(0.25, 100).unwrap();
        assert_eq!(bisect(0_u64, 100, 1, |k| binomial.cdf(k) - 0.5), Some(24));
        let root = bisect(10.0, 0.0, 1e-6, |x: f64| x * x - 9.0).unwrap();
        assert_float_eq(root, 3.0);
    }

    #[test]
    fn bisect_needs_a_sign_change() {
        assert_eq!(bisect(4.0, 10.0, 1e-6, |x: f64| x * x - 9.0), None);
        assert_eq!(bisect(0.0, 10.0, 1e-6, |_| f64::NAN), None);
        assert_eq!(bisect(3.0, 10.0, 1e-6, |x: f64| x * x - 9.0), Some(3.0));
    }

    #[test]
    fn bisect_stops_at_adjacent_floats() {
        let root = bisect(0.0, 10.0, 0.0, |x: f64| x * x - 2.0).unwrap();
        assert_close(root, 2.0_f64.sqrt());
    }
}
