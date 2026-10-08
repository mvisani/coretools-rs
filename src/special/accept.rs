//! Metropolis–Hastings acceptance decisions that skip the logarithm of the uniform draw whenever a
//! lookup table already decides them.

use rand::{Rng, RngExt};

use super::logit;
use crate::probability::LinearProbability;

/// Number of bins of a [`Lookup`].
const BINS: usize = 100;

/// The tables hold an increasing `f` at the bin edges rounded to 11 decimals (off by at most 5e-12);
/// widening every bound by this margin keeps it on the safe side of `f(x)`.
const MARGIN: f64 = 1e-9;

/// An increasing function `f` tabulated at the edges of `BINS` equal bins of `[min_x, max_x]`.
struct Lookup {
    min_x: f64,
    max_x: f64,
    inv_bin_width: f64,
    bounds: [f64; BINS + 1],
}

impl Lookup {
    /// Decides `f(x) <= q`, evaluating `f` only if `q` lies within the bounds of the bin of `x`.
    fn decide(&self, x: f64, q: f64, f: impl FnOnce(f64) -> f64) -> bool {
        let (lower, upper) = if x < self.min_x {
            (f64::NEG_INFINITY, self.bounds[0])
        } else if x > self.max_x {
            (self.bounds[BINS], f64::INFINITY)
        } else {
            // (x - min_x) * inv_bin_width is in [0, BINS]: the cast truncates to the bin as intended;
            // x = max_x belongs to the last bin.
            let bin = (((x - self.min_x) * self.inv_bin_width) as usize).min(BINS - 1);
            (self.bounds[bin], self.bounds[bin + 1])
        };
        // NaN fails both comparisons and reaches the exact check, as in the naive one.
        if q >= upper + MARGIN {
            true
        } else if q < lower - MARGIN {
            false
        } else {
            f(x) <= q
        }
    }
}

/// `ln` on `[1e-10, 1]`, from C++ coretools `TAccept.h`. (`BINS as f64`: 100 is exact.)
#[rustfmt::skip]
const LN: Lookup = Lookup {
    min_x: 1e-10,
    max_x: 1.0,
    inv_bin_width: BINS as f64 / (1.0 - 1e-10),
    bounds: [
        -23.02585092994, -4.60517017609, -3.91202300053, -3.50655789409, -3.21887582247, -2.99573227165,
        -2.81341071519, -2.65926003560, -2.52572864316, -2.40794560764, -2.30258509209, -2.20727491238,
        -2.12026353547, -2.04022082786, -1.96611285576, -1.89711998432, -1.83258146322, -1.77195684144,
        -1.71479842764, -1.66073120640, -1.60943791203, -1.56064774789, -1.51412773228, -1.46967596972,
        -1.42711635532, -1.38629436082, -1.34707364768, -1.30933331971, -1.27296567556, -1.23787435576,
        -1.20397280409, -1.17118298128, -1.13943428298, -1.10866262432, -1.07880966118, -1.04982212431,
        -1.02165124735, -0.99425227317, -0.96758402610, -0.94160853970, -0.91629073172, -0.89159811914,
        -0.86750056757, -0.84397007016, -0.82098055194, -0.79850769610, -0.77652878938, -0.75502258417,
        -0.73396917497, -0.71334988777, -0.69314718046, -0.67334455317, -0.65392646731, -0.63487827235,
        -0.61618613934, -0.59783700067, -0.57981849517, -0.56211891808, -0.54472717537, -0.52763274201,
        -0.51082562370, -0.49429632175, -0.47803580088, -0.46203545954, -0.44628710257, -0.43078291604,
        -0.41551544391, -0.40047756655, -0.38566248076, -0.37106368135, -0.35667494390, -0.34249030891,
        -0.32850406693, -0.31471074480, -0.30110509275, -0.28768207242, -0.27443684567, -0.26136476410,
        -0.24846135927, -0.23572233349, -0.22314355129, -0.21072103129, -0.19845093870, -0.18632957817,
        -0.17435338713, -0.16251892948, -0.15082288972, -0.13926206732, -0.12783337150, -0.11653381624,
        -0.10536051565, -0.09431067946, -0.08338160893, -0.07257069283, -0.06187540371, -0.05129329438,
        -0.04082199452, -0.03045920748, -0.02020270732, -0.01005033585, 0.00000000000,
    ],
};

/// `logit` on `[1e-10, 1 - 1e-10]`, from C++ coretools `TAcceptOddsRation.h`. (`BINS as f64`: 100 is
/// exact.)
#[rustfmt::skip]
const LOGIT: Lookup = Lookup {
    min_x: 1e-10,
    max_x: 1.0 - 1e-10,
    inv_bin_width: BINS as f64 / ((1.0 - 1e-10) - 1e-10),
    bounds: [
        -23.02585092984, -4.59511984024, -3.89182029321, -3.47609868661, -3.17805382795, -2.94443897727,
        -2.75153531148, -2.58668934278, -2.44234703423, -2.31363492818, -2.19722457645, -2.09074109614,
        -1.99243016397, -1.90095876054, -1.81528996604, -1.73460105484, -1.65822807610, -1.58562726327,
        -1.51634748893, -1.45001017510, -1.38629436074, -1.32492541439, -1.26566637300, -1.20831120562,
        -1.15267950965, -1.09861228840, -1.04596855493, -0.99462257491, -0.94446160862, -0.89538404685,
        -0.84729786020, -0.80011929993, -0.75377180221, -0.70818505777, -0.66329421727, -0.61903920827,
        -0.57536414478, -0.53221681364, -0.48954822522, -0.44731221795, -0.40546510802, -0.36396537713,
        -0.32277339220, -0.28185115208, -0.24116205677, -0.20067069542, -0.16034265004, -0.12014431182,
        -0.08004270766, -0.04000533461, 0.00000000000, 0.04000533461, 0.08004270766, 0.12014431182,
        0.16034265004, 0.20067069542, 0.24116205677, 0.28185115208, 0.32277339220, 0.36396537713,
        0.40546510802, 0.44731221795, 0.48954822522, 0.53221681364, 0.57536414478, 0.61903920827,
        0.66329421727, 0.70818505777, 0.75377180221, 0.80011929993, 0.84729786020, 0.89538404685,
        0.94446160862, 0.99462257491, 1.04596855493, 1.09861228840, 1.15267950965, 1.20831120562,
        1.26566637300, 1.32492541439, 1.38629436074, 1.45001017510, 1.51634748893, 1.58562726327,
        1.65822807610, 1.73460105484, 1.81528996604, 1.90095876054, 1.99243016397, 2.09074109614,
        2.19722457645, 2.31363492818, 2.44234703423, 2.58668934278, 2.75153531148, 2.94443897727,
        3.17805382795, 3.47609868661, 3.89182029321, 4.59511984024, 23.02585084710,
    ],
};

/// The Metropolis–Hastings acceptance of a proposal with acceptance ratio `q`, given `ln q`.
///
/// Draws one `u = rng.random::<f64>()` and returns exactly `u.ln() <= log_q` (so `log_q ≥ 0`
/// always accepts and NaN never does), but computes `u.ln()` only when a lookup table of `ln`
/// cannot decide.
pub fn accept_log<R: Rng + ?Sized>(rng: &mut R, log_q: f64) -> bool {
    let u: f64 = rng.random();
    LN.decide(u, log_q, f64::ln)
}

/// The acceptance of a proposal with acceptance probability `q`, given its log-odds
/// `logit(q) = ln(q / (1 - q))`.
///
/// Draws one `u = rng.random::<f64>()` and returns exactly [`logit`]`(u) <= logit_q`, but computes
/// the logit only when a lookup table of it cannot decide.
pub fn accept_logit<R: Rng + ?Sized>(rng: &mut R, logit_q: f64) -> bool {
    let u: f64 = rng.random();
    LOGIT.decide(u, logit_q, |u| logit(LinearProbability::new_unchecked(u)))
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use rand::{RngExt, SeedableRng, TryRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// A generator whose `random::<f64>()` is a chosen `x`.
    struct Draws(f64);

    impl TryRng for Draws {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Infallible> {
            unreachable!("random::<f64>() draws a u64")
        }

        fn try_next_u64(&mut self) -> Result<u64, Infallible> {
            // random::<f64>() is (u64 >> 11) * 2^-53; x is a multiple of 2^-53 in [0, 1): lossless
            let k = self.0 * 2f64.powi(53);
            assert!(k.fract() == 0.0 && (0.0..2f64.powi(53)).contains(&k));
            Ok((k as u64) << 11)
        }

        fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Infallible> {
            unreachable!("random::<f64>() draws a u64")
        }
    }

    /// Multiples of 2^-53 (the values `random::<f64>()` takes) at and around `x`.
    fn draws_around(x: f64) -> impl Iterator<Item = f64> {
        let scale = 2f64.powi(53);
        // x in [0, 1], so k and the kept neighbours lie in [0, 2^53]: both casts are exact
        let k = (x * scale).floor() as i64;
        (k - 2..=k + 2)
            .filter(|&k| (0..1 << 53).contains(&k))
            .map(move |k| k as f64 / scale)
    }

    /// Ratios at and around `boundary` (within and beyond the lookup tables' rounding of 5e-12), at
    /// and next to `exact`, and the extremes.
    fn ratios_around(boundary: f64, exact: f64) -> Vec<f64> {
        let mut ratios = vec![0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        for d in [2e-9, 1e-9, 1e-11, 5e-12, 1e-12, 0.0] {
            ratios.extend([boundary + d, boundary - d]);
        }
        ratios.extend([exact, exact.next_up(), exact.next_down()]);
        ratios
    }

    // --- ported from C++ TAcceptUnitTests.cpp ---------------------------------

    #[test]
    fn log_scale_decisions_equal_the_naive_check_for_a_seeded_generator() {
        let mut ratios = ChaCha8Rng::seed_from_u64(5);
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let mut naive_rng = rng.clone();
        for _ in 0..1_000_000 {
            let log_q = ratios.random::<f64>().ln();
            let naive = naive_rng.random::<f64>().ln() <= log_q;
            assert_eq!(accept_log(&mut rng, log_q), naive, "ln q = {log_q}");
        }
    }

    // --- ported from C++ TAcceptOddsRatioTests.cpp ----------------------------

    fn naive_logit(u: f64) -> f64 {
        logit(LinearProbability::new(u).unwrap())
    }

    #[test]
    fn logit_scale_decisions_equal_the_naive_check_for_a_seeded_generator() {
        let mut ratios = ChaCha8Rng::seed_from_u64(7);
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let mut naive_rng = rng.clone();
        for _ in 0..1_000_000 {
            let logit_q = naive_logit(ratios.random());
            let naive = naive_logit(naive_rng.random()) <= logit_q;
            assert_eq!(
                accept_logit(&mut rng, logit_q),
                naive,
                "logit q = {logit_q}"
            );
        }
    }

    // --- beyond the C++ tests -----------------------------------------------

    /// Draws at and around each edge and the extremes of `[0, 1)`, against ratios around `f(edge)`
    /// and `f(x)`: `accept` decides as the naive `f(x) <= q`.
    fn assert_naive_around(
        edges: impl Iterator<Item = f64>,
        f: impl Fn(f64) -> f64,
        accept: impl Fn(&mut Draws, f64) -> bool,
    ) {
        let extremes = [0.0, 1e-300, 1.0 - 2f64.powi(-53)];
        for edge in edges.chain(extremes) {
            for x in draws_around(edge) {
                for q in ratios_around(f(edge), f(x)) {
                    assert_eq!(accept(&mut Draws(x), q), f(x) <= q, "x = {x}, q = {q}");
                }
            }
        }
    }

    #[test]
    fn log_scale_decisions_equal_the_naive_check_around_every_bin_edge() {
        // The lookup table has 100 bins of width (1 - 1e-10) / 100 from 1e-10.
        let edges = (0..=100).map(|i| 1e-10 + f64::from(i) * (1.0 - 1e-10) / 100.0);
        assert_naive_around(edges, f64::ln, accept_log);
    }

    #[test]
    fn log_scale_draw_just_below_a_bin_edge_is_rejected_above_ln_q() {
        // ln q lies between the table's 11-decimal ln of the edge (-4.60517017609) and ln x: C++
        // accepted from the table.
        let x: f64 = 0.010_000_000_098_999_928;
        let log_q = -4.605_170_176_089;
        assert!(x.ln() > log_q);
        assert!(!accept_log(&mut Draws(x), log_q));
    }

    #[test]
    fn log_scale_decides_positive_and_nan_ratios_as_the_naive_check() {
        // C++ asserted ln q <= 0 in debug builds and accepted NaN for draws below 1e-10.
        assert!(accept_log(&mut Draws(0.5), 0.3));
        assert!(!accept_log(&mut Draws(0.0), f64::NAN));
        assert!(!accept_log(&mut Draws(0.5), f64::NAN));
    }

    #[test]
    fn logit_scale_decisions_equal_the_naive_check_around_every_bin_edge() {
        // The lookup table has 100 bins of width (1 - 2e-10) / 100 from 1e-10.
        let edges = (0..=100).map(|i| 1e-10 + f64::from(i) * (1.0 - 2e-10) / 100.0);
        assert_naive_around(edges, naive_logit, |rng, logit_q| {
            accept_logit(rng, logit_q)
        });
    }

    #[test]
    fn logit_scale_draw_just_below_a_bin_edge_is_rejected_above_logit_q() {
        // logit q lies between the table's 11-decimal logit of the edge (-4.59511984024) and
        // logit x: C++ accepted from the table.
        let x = 0.010_000_000_097_999_95;
        let logit_q = -4.595_119_840_238;
        assert!(naive_logit(x) > logit_q);
        assert!(!accept_logit(&mut Draws(x), logit_q));
    }

    #[test]
    fn logit_scale_rejects_nan_ratios() {
        // C++ accepted NaN for draws below 1e-10.
        assert!(!accept_logit(&mut Draws(0.0), f64::NAN));
        assert!(!accept_logit(&mut Draws(0.5), f64::NAN));
        assert!(!accept_logit(&mut Draws(1.0 - 2f64.powi(-53)), f64::NAN));
    }
}
