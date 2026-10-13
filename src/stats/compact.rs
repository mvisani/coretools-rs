//! Byte-sized encodings of large counts.

/// A count stored as a power of two in one byte: `⌈log2 n⌉`, so `n` is rounded up to the next power
/// of two (3 becomes 4). Holds 1 to 2^63.
///
/// ```
/// use coretools_rs::stats::LogInt;
///
/// let count = LogInt::from_linear(1000);
/// assert_eq!(count.log(), 10);
/// assert_eq!(count.linear(), 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogInt(u8);

impl LogInt {
    /// 1.
    pub const MIN: Self = Self(0);
    /// 2^63.
    pub const MAX: Self = Self(63);

    /// `⌈log2 linear⌉`; counts above 2^63 saturate at [`LogInt::MAX`].
    ///
    /// # Panics
    ///
    /// If `linear` is 0, which has no logarithm.
    pub const fn from_linear(linear: u64) -> Self {
        assert!(linear > 0, "a LogInt encodes counts of at least 1");
        // ⌈log2 n⌉ = number of bits of n - 1, for n ≥ 1
        let log = u64::BITS - (linear - 1).leading_zeros();
        if log > Self::MAX.0 as u32 {
            Self::MAX
        } else {
            // log ≤ 63 fits a byte
            Self(log as u8)
        }
    }

    /// The count `2^log`.
    ///
    /// # Panics
    ///
    /// If `log` is above 63: `2^64` does not fit a `u64`.
    pub fn from_log(log: u8) -> Self {
        assert!(
            log <= Self::MAX.0,
            "a LogInt holds logs up to 63, not {log}"
        );
        Self(log)
    }

    pub const fn log(self) -> u8 {
        self.0
    }

    pub const fn linear(self) -> u64 {
        1 << self.0
    }
}

/// A count stored in one byte: exact up to [`PseudoInt::MAX_EXACT`] (192), log-scaled above, up to
/// `2^63 + 255`.
///
/// A count `n > 192` is stored as `p = 192 + ⌊log2(n - 192)⌋`, and the byte `p > 192` stands for the
/// count `2^(p - 192) + p`.
///
/// ```
/// use coretools_rs::stats::PseudoInt;
///
/// assert_eq!(PseudoInt::from_linear(150).linear(), 150);
/// let count = PseudoInt::from_linear(1000);
/// assert_eq!(count.pseudo(), 201);
/// assert_eq!(count.linear(), 713);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PseudoInt(u8);

impl PseudoInt {
    /// The largest count stored exactly.
    pub const MAX_EXACT: u64 = 255 - 63;
    /// 0.
    pub const MIN: Self = Self(0);
    /// `2^63 + 255`.
    pub const MAX: Self = Self(255);

    /// Encodes `linear`; every `u64` fits, counts from `2^63 + 192` on being [`PseudoInt::MAX`].
    pub const fn from_linear(linear: u64) -> Self {
        if linear <= Self::MAX_EXACT {
            // ≤ 192 fits a byte
            Self(linear as u8)
        } else {
            // ilog2 of a u64 is at most 63, so the byte is at most 255
            Self((Self::MAX_EXACT as u32 + (linear - Self::MAX_EXACT).ilog2()) as u8)
        }
    }

    pub const fn from_pseudo(pseudo: u8) -> Self {
        Self(pseudo)
    }

    pub const fn pseudo(self) -> u8 {
        self.0
    }

    pub const fn linear(self) -> u64 {
        let pseudo = self.0 as u64;
        if pseudo <= Self::MAX_EXACT {
            pseudo
        } else {
            (1 << (pseudo - Self::MAX_EXACT)) + pseudo
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_int_extremes() {
        assert_eq!(LogInt::MIN, LogInt::from_linear(1));
        assert_eq!(LogInt::MIN, LogInt::from_log(0));
        assert_eq!(LogInt::MAX, LogInt::from_linear(1 << 63));
        assert_eq!(LogInt::MAX, LogInt::from_log(63));
        assert_eq!(LogInt::MAX.linear(), 1 << 63);
    }

    #[test]
    fn log_int_rounds_up_to_a_power_of_two() {
        let li = LogInt::from_linear(1);
        assert_eq!((li.log(), li.linear()), (0, 1));
        let li = LogInt::from_log(1);
        assert_eq!((li.log(), li.linear()), (1, 2));
        let li = LogInt::from_linear(3);
        assert_eq!((li.log(), li.linear()), (2, 4));

        for log in 3..64 {
            let from_log = LogInt::from_log(log);
            let from_linear = LogInt::from_linear(1 << log);
            let from_just_below = LogInt::from_linear((1 << log) - 1);
            assert_eq!(from_log.log(), log);
            assert_eq!(from_log.linear(), 1 << log);
            assert_eq!(from_linear, from_log);
            assert_eq!(from_just_below, from_log);
        }
    }

    #[test]
    fn log_int_rounds_up_just_above_large_powers_of_two() {
        // 2^53 + 1 is not an f64; the exact ceiling is 54
        assert_eq!(LogInt::from_linear((1 << 53) + 1).log(), 54);
        assert_eq!(LogInt::from_linear((1 << 62) + 1).log(), 63);
    }

    #[test]
    fn log_int_saturates_above_its_largest_power_of_two() {
        assert_eq!(LogInt::from_linear((1 << 63) + 1), LogInt::MAX);
        assert_eq!(LogInt::from_linear(u64::MAX), LogInt::MAX);
    }

    #[test]
    #[should_panic(expected = "a LogInt encodes counts of at least 1")]
    fn log_int_of_zero_is_a_dev_error() {
        LogInt::from_linear(0);
    }

    #[test]
    #[should_panic(expected = "a LogInt holds logs up to 63, not 64")]
    fn log_int_log_beyond_63_is_a_dev_error() {
        LogInt::from_log(64);
    }

    #[test]
    fn pseudo_int_extremes() {
        assert_eq!(PseudoInt::MIN, PseudoInt::from_linear(0));
        assert_eq!(PseudoInt::MIN, PseudoInt::from_pseudo(0));
        assert_eq!(PseudoInt::MAX, PseudoInt::from_linear((1 << 63) + 255));
        assert_eq!(PseudoInt::MAX, PseudoInt::from_pseudo(255));
        assert_eq!(PseudoInt::MAX.linear(), (1 << 63) + 255);
    }

    #[test]
    fn pseudo_int_is_exact_up_to_192() {
        for linear in 0..=PseudoInt::MAX_EXACT {
            let pi = PseudoInt::from_linear(linear);
            assert_eq!(u64::from(pi.pseudo()), linear);
            assert_eq!(pi.linear(), linear);
        }
    }

    #[test]
    fn pseudo_int_round_trips_above_192() {
        for pseudo in 193..=255 {
            let from_pseudo = PseudoInt::from_pseudo(pseudo);
            assert_eq!(PseudoInt::from_linear(from_pseudo.linear()), from_pseudo);
            assert_eq!(
                PseudoInt::from_linear(from_pseudo.linear() - 1),
                from_pseudo
            );
        }
    }

    #[test]
    fn pseudo_int_takes_the_exact_log_just_below_large_powers_of_two() {
        // 192 + 2^60 - 1: log2(2^60 - 1) rounds to 60 in f64, its floor is 59
        assert_eq!(
            PseudoInt::from_linear(192 + (1 << 60) - 1).pseudo(),
            192 + 59
        );
    }

    #[test]
    fn pseudo_int_encodes_every_u64() {
        assert_eq!(PseudoInt::from_linear(u64::MAX), PseudoInt::MAX);
    }
}
