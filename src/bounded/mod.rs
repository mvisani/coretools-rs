//! Bounded values: numbers whose interval is part of their type, and numeric ranges.
//!
//! [`Bounded<T, I>`] stores a `T` that lies in the interval named by the marker `I` (see [`interval`]);
//! the aliases ([`Positive`], [`ZeroOneOpen`], [`UnsignedInt8`], …) cover the common cases.
//! [`Bounded::new`] and [`FromStr`] check the interval and name it in the error;
//! [`Bounded::new_unchecked`] only checks in debug builds.
//!
//! Arithmetic between two values of the same type returns a `Bounded` value (and has a compound
//! assignment such as `+=`) where the result always lies in the interval, and the raw number otherwise:
//!
//! | interval | `+` | `-` | `*` | `/` |
//! |---|---|---|---|---|
//! | `f64` [`Unbounded`] | bounded | bounded | bounded | bounded |
//! | `f64` [`Positive`], [`StrictlyPositive`] | bounded | raw | bounded | bounded |
//! | `f64` [`Negative`], [`StrictlyNegative`] | bounded | raw | raw | raw |
//! | `f64` [`ZeroOneClosed`], [`ZeroOneOpen`], [`ZeroOpenOneClosed`] | raw | raw | bounded | raw |
//! | unsigned [`interval::Positive`] | bounded | bounded | bounded | bounded |
//! | unsigned [`interval::StrictlyPositive`] | bounded | raw | bounded | raw |
//!
//! Negation returns an [`Unbounded`] for [`Unbounded`] and the raw number otherwise. For `f64`, an
//! overflow, underflow or division by zero that carries a "bounded" result out of the interval is caught
//! by a debug assertion only. For unsigned integers, overflow, subtraction below 0 and division by zero
//! always panic.
//!
//! [`NumericRange`] is a range of numbers with each end included or excluded, parsed from and printed
//! in the C++ coretools syntax (`[a,b)`, `]a,b[`, `a,b`, `-inf`, `inf`).

pub mod interval;
mod ops;
mod range;

pub use range::NumericRange;

use core::cmp::Ordering;
use core::fmt;
use core::marker::PhantomData;
use core::num::{IntErrorKind, ParseFloatError, ParseIntError};
use core::str::FromStr;

use num_traits::ToPrimitive;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a string is not a number of the expected type.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseNumberError {
    #[error(transparent)]
    Float(#[from] ParseFloatError),

    #[error(transparent)]
    Int(#[from] ParseIntError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BoundedError {
    #[error("{value} is outside the interval {interval}")]
    OutOfInterval {
        interval: &'static str,
        value: String,
    },

    /// A string read as a bounded value is not a number.
    #[error("cannot parse {input:?} as a number in {interval}")]
    Parse {
        interval: &'static str,
        input: String,
        #[source]
        source: ParseNumberError,
    },

    /// A bound of a [`NumericRange`] of raw numbers is not a number.
    #[error("not a number")]
    NotANumber(#[source] ParseNumberError),

    #[error("range minimum {min} is above its maximum {max}")]
    MinAboveMax { min: String, max: String },

    #[error("invalid range {input:?}: {reason}")]
    InvalidRange { input: String, reason: &'static str },

    #[error("invalid bound in range {input:?}")]
    InvalidBound {
        input: String,
        #[source]
        source: Box<BoundedError>,
    },
}

impl From<ParseFloatError> for BoundedError {
    fn from(e: ParseFloatError) -> Self {
        Self::NotANumber(e.into())
    }
}

impl From<ParseIntError> for BoundedError {
    fn from(e: ParseIntError) -> Self {
        Self::NotANumber(e.into())
    }
}

// ---------------------------------------------------------------------------
// Interval
// ---------------------------------------------------------------------------

/// An interval of `T`, carried by a zero-sized marker type (see [`interval`]).
pub trait Interval<T>: Copy + Default + fmt::Debug + 'static {
    /// Smallest value inside the interval.
    const MIN: T;
    /// Largest value inside the interval.
    const MAX: T;
    /// The interval in mathematical notation, e.g. `(0, 1)`; used in error messages.
    const NOTATION: &'static str;

    /// Whether `value` lies inside the interval (`false` for NaN).
    #[inline]
    fn contains(value: T) -> bool
    where
        T: PartialOrd,
    {
        value >= Self::MIN && value <= Self::MAX
    }
}

// ---------------------------------------------------------------------------
// Bounded
// ---------------------------------------------------------------------------

/// A `T` guaranteed to lie in the interval `I`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bounded<T, I> {
    value: T,
    _interval: PhantomData<I>,
}

impl<T: Copy + PartialOrd + fmt::Display, I: Interval<T>> Bounded<T, I> {
    /// Smallest value of the interval.
    pub const MIN: Self = Self::from_raw(I::MIN);
    /// Largest value of the interval.
    pub const MAX: Self = Self::from_raw(I::MAX);

    const fn from_raw(value: T) -> Self {
        Self {
            value,
            _interval: PhantomData,
        }
    }

    /// Checked constructor.
    pub fn new(value: T) -> Result<Self, BoundedError> {
        if I::contains(value) {
            Ok(Self::from_raw(value))
        } else {
            Err(BoundedError::OutOfInterval {
                interval: I::NOTATION,
                value: value.to_string(),
            })
        }
    }

    /// Constructor for values known to lie in the interval; checks only in debug builds.
    /// Not `unsafe`: a value outside the interval is a logic error, not undefined behaviour.
    #[inline]
    #[track_caller]
    pub fn new_unchecked(value: T) -> Self {
        debug_assert!(
            I::contains(value),
            "{value} is outside the interval {}",
            I::NOTATION
        );
        Self::from_raw(value)
    }

    #[inline]
    pub const fn get(self) -> T {
        self.value
    }
}

/// The smallest value of the interval.
impl<T: Copy + PartialOrd + fmt::Display, I: Interval<T>> Default for Bounded<T, I> {
    fn default() -> Self {
        Self::MIN
    }
}

impl<T: fmt::Display, I> fmt::Display for Bounded<T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

/// Parses the number, ignoring surrounding whitespace, and checks the interval. An integer too large
/// (or negative) for `T` is reported as outside the interval.
impl<T, I> FromStr for Bounded<T, I>
where
    T: Copy + PartialOrd + fmt::Display + FromStr<Err: Into<ParseNumberError>>,
    I: Interval<T>,
{
    type Err = BoundedError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        match trimmed.parse::<T>() {
            Ok(value) => Self::new(value),
            Err(e) => Err(match e.into() {
                ParseNumberError::Int(e)
                    if matches!(
                        e.kind(),
                        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow
                    ) || trimmed.parse::<i128>().is_ok() =>
                {
                    BoundedError::OutOfInterval {
                        interval: I::NOTATION,
                        value: trimmed.to_owned(),
                    }
                }
                source => BoundedError::Parse {
                    interval: I::NOTATION,
                    input: s.to_owned(),
                    source,
                },
            }),
        }
    }
}

/// The interval's ends, so that e.g. a [`NumericRange`] of bounded values defaults to the interval.
impl<T: Copy + PartialOrd + fmt::Display, I: Interval<T>> num_traits::Bounded for Bounded<T, I> {
    fn min_value() -> Self {
        Self::MIN
    }
    fn max_value() -> Self {
        Self::MAX
    }
}

impl<T: ToPrimitive, I> ToPrimitive for Bounded<T, I> {
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        self.value.to_f64()
    }
}

// --- comparison against the raw value ----------------------------------------
// Implemented per value type: a blanket `PartialEq<T>` would overlap with the
// derived `PartialEq<Self>`.

macro_rules! cmp_raw {
    ($($t:ty),*) => {$(
        impl<I> PartialEq<$t> for Bounded<$t, I> {
            fn eq(&self, other: &$t) -> bool {
                self.value == *other
            }
        }
        impl<I> PartialOrd<$t> for Bounded<$t, I> {
            fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
                self.value.partial_cmp(other)
            }
        }
    )*};
}
cmp_raw!(f64, u8, u16, u32, u64, usize);

// ---------------------------------------------------------------------------
// Aliases (the C++ common weak types)
// ---------------------------------------------------------------------------

/// Any finite `f64`.
pub type Unbounded = Bounded<f64, interval::Unbounded>;
/// `f64` in `[0, 1]`.
pub type ZeroOneClosed = Bounded<f64, interval::ZeroOneClosed>;
/// `f64` in `(0, 1)`.
pub type ZeroOneOpen = Bounded<f64, interval::ZeroOneOpen>;
/// `f64` in `(0, 1]`.
pub type ZeroOpenOneClosed = Bounded<f64, interval::ZeroOpenOneClosed>;
/// `f64` in `[0, inf)`.
pub type Positive = Bounded<f64, interval::Positive>;
/// `f64` in `(0, inf)`.
pub type StrictlyPositive = Bounded<f64, interval::StrictlyPositive>;
/// `f64` in `(-inf, 0]`.
pub type Negative = Bounded<f64, interval::Negative>;
/// `f64` in `(-inf, 0)`.
pub type StrictlyNegative = Bounded<f64, interval::StrictlyNegative>;
/// `usize` above 0.
pub type StrictlyPositiveUInt = Bounded<usize, interval::StrictlyPositive>;
/// Any `u8`.
pub type UnsignedInt8 = Bounded<u8, interval::Positive>;
/// Any `u16`.
pub type UnsignedInt16 = Bounded<u16, interval::Positive>;
/// Any `u32`.
pub type UnsignedInt32 = Bounded<u32, interval::Positive>;
/// Any `u64`.
pub type UnsignedInt64 = Bounded<u64, interval::Positive>;

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_close;

    #[test]
    fn checked_construction_names_the_interval() {
        assert!(Positive::new(-0.1).is_err());
        assert!(Positive::new(0.0).is_ok());
        assert!(Positive::new(10002.348321).is_ok());

        let err = ZeroOneOpen::new(1.0).unwrap_err();
        assert!(matches!(
            err,
            BoundedError::OutOfInterval {
                interval: "(0, 1)",
                ..
            }
        ));
        assert_eq!(err.to_string(), "1 is outside the interval (0, 1)");

        let err = "1.5".parse::<ZeroOneOpen>().unwrap_err();
        assert_eq!(err.to_string(), "1.5 is outside the interval (0, 1)");
        let err = "abc".parse::<Positive>().unwrap_err();
        assert!(matches!(
            err,
            BoundedError::Parse {
                interval: "[0, inf)",
                source: ParseNumberError::Float(_),
                ..
            }
        ));
        assert_eq!(
            err.to_string(),
            r#"cannot parse "abc" as a number in [0, inf)"#
        );
        assert_eq!(" 14.31 ".parse::<Positive>().unwrap(), 14.31);
    }

    #[test]
    fn integers_that_do_not_fit_the_type_are_outside_the_interval() {
        let err = "256".parse::<UnsignedInt8>().unwrap_err();
        assert_eq!(err.to_string(), "256 is outside the interval [0, 255]");
        let err = " -1 ".parse::<UnsignedInt8>().unwrap_err();
        assert_eq!(err.to_string(), "-1 is outside the interval [0, 255]");
        let huge = "1".repeat(50);
        assert!(matches!(
            huge.parse::<UnsignedInt64>(),
            Err(BoundedError::OutOfInterval { .. })
        ));
        let err = "1.5".parse::<UnsignedInt8>().unwrap_err();
        assert!(matches!(
            err,
            BoundedError::Parse {
                source: ParseNumberError::Int(_),
                ..
            }
        ));
    }

    #[test]
    fn unsigned_notation_names_the_type_maximum() {
        fn check<T: fmt::Display>()
        where
            interval::Positive: Interval<T>,
            interval::StrictlyPositive: Interval<T>,
        {
            let max = <interval::Positive as Interval<T>>::MAX;
            assert_eq!(
                <interval::Positive as Interval<T>>::NOTATION,
                format!("[0, {max}]")
            );
            assert_eq!(
                <interval::StrictlyPositive as Interval<T>>::NOTATION,
                format!("[1, {max}]")
            );
        }
        check::<u8>();
        check::<u16>();
        check::<u32>();
        check::<u64>();
        check::<usize>();
    }

    #[test]
    fn display_prints_the_shortest_round_trip_representation() {
        let v = Positive::new(0.1234567).unwrap();
        assert_eq!(v.to_string(), "0.1234567");
        assert_eq!(v.to_string().parse::<Positive>().unwrap(), v);
        assert_eq!(UnsignedInt8::new(255).unwrap().to_string(), "255");
    }

    #[track_caller]
    fn assert_admits<I: Interval<f64>>(inside: &[f64], outside: &[f64]) {
        for &v in inside {
            assert!(
                Bounded::<f64, I>::new(v).is_ok(),
                "{v} rejected by {}",
                I::NOTATION
            );
        }
        for &v in outside {
            assert!(
                Bounded::<f64, I>::new(v).is_err(),
                "{v} admitted by {}",
                I::NOTATION
            );
        }
    }

    #[test]
    fn each_interval_admits_exactly_its_finite_values() {
        use interval::*;
        const INF: f64 = f64::INFINITY;
        const NAN: f64 = f64::NAN;
        assert_admits::<Unbounded>(&[f64::MIN, -0.1, 0.0, 0.1, f64::MAX], &[-INF, INF, NAN]);
        assert_admits::<ZeroOneClosed>(&[0.0, 0.5, 1.0], &[-0.1, 1.1, NAN]);
        assert_admits::<ZeroOneOpen>(&[0.1, 0.99], &[-0.1, 0.0, 1.0, 1.2, NAN]);
        assert_admits::<ZeroOpenOneClosed>(&[0.1, 1.0], &[0.0, 1.0 + f64::EPSILON, NAN]);
        assert_admits::<Positive>(&[0.0, 10002.348321, f64::MAX], &[-0.1, INF, NAN]);
        assert_admits::<StrictlyPositive>(&[f64::MIN_POSITIVE, 10002.348321], &[-0.1, 0.0, INF]);
        assert_admits::<Negative>(&[f64::MIN, -0.1, 0.0], &[0.1, -INF, NAN]);
        assert_admits::<StrictlyNegative>(&[f64::MIN, -0.1], &[0.0, 0.1, -INF, NAN]);
    }

    #[test]
    fn open_float_ends_exclude_only_the_end_point() {
        let tiny = f64::from_bits(1); // smallest positive subnormal
        let below_one = 1.0 - f64::EPSILON / 2.0;
        assert!(StrictlyPositive::new(tiny).is_ok());
        assert!(StrictlyNegative::new(-tiny).is_ok());
        assert!(ZeroOneOpen::new(tiny).is_ok());
        assert!(ZeroOneOpen::new(below_one).is_ok());
        assert!(ZeroOpenOneClosed::new(tiny).is_ok());
        assert_eq!(StrictlyPositive::MIN, tiny);
        assert_eq!(ZeroOneOpen::MAX, below_one);
    }

    #[test]
    fn strictly_positive_integers_start_at_one() {
        assert!(StrictlyPositiveUInt::new(0).is_err());
        assert_eq!(StrictlyPositiveUInt::MIN, 1);
        assert_eq!(Bounded::<u8, interval::StrictlyPositive>::MIN, 1);
        assert_eq!(UnsignedInt8::MIN, 0);
        assert_eq!(UnsignedInt8::MAX, 255);
        assert_eq!(UnsignedInt8::default(), 0);
        let err = "0"
            .parse::<Bounded<u8, interval::StrictlyPositive>>()
            .unwrap_err();
        assert_eq!(err.to_string(), "0 is outside the interval [1, 255]");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "-0.1 is outside the interval [0, inf)")]
    fn unchecked_construction_outside_the_interval_is_a_dev_error() {
        Positive::new_unchecked(-0.1);
    }

    // --- ported from C++ commonWeakTypesUnitTests.cpp ------------------------
    // Arithmetic that keeps a value in its interval returns a `Bounded`, other
    // arithmetic returns the raw number; the `let x: T` annotations pin which.
    // Not ported: the `IntervalTypes` tests, which pair intervals with arbitrary
    // Check/NoCheck skills (each interval now has fixed arithmetic, exercised
    // below), `UnsignedIntWithMax` (mutable maxima are dropped), and arithmetic
    // with a bare number (callers use `get()`).

    fn positive(v: f64) -> Positive {
        Positive::new(v).unwrap()
    }
    fn strictly_positive(v: f64) -> StrictlyPositive {
        StrictlyPositive::new(v).unwrap()
    }
    fn zero_one_open(v: f64) -> ZeroOneOpen {
        ZeroOneOpen::new(v).unwrap()
    }
    fn unbounded(v: f64) -> Unbounded {
        Unbounded::new(v).unwrap()
    }

    #[test]
    fn positive_arithmetic() {
        let a = positive(0.1);
        let b = positive(0.5);
        assert_eq!(a.get(), 0.1);
        assert_eq!("14.31".parse::<Positive>().unwrap(), 14.31);

        assert_ne!(a, b);
        assert!(a < b);
        assert!(a != 1.5);
        assert!(a < 1.5);

        let mut c: Positive = a * a;
        assert_close(c.get(), 0.01);
        c /= a;
        assert_close(c.get(), 0.1);
        let c: Positive = a / b;
        assert_close(c.get(), 0.2);
        let mut c: Positive = a + b;
        assert_close(c.get(), 0.6);
        c += b;
        assert_close(c.get(), 1.1);
        c *= b;
        assert_close(c.get(), 0.55);

        let d: f64 = b - a;
        assert_close(d, 0.4);
        assert!(Positive::new(a - b).is_err());
        let d: f64 = -b;
        assert_eq!(d, -0.5);
    }

    #[test]
    fn strictly_positive_arithmetic() {
        assert!(StrictlyPositive::new(0.0).is_err());
        let a = strictly_positive(0.1);
        let b = strictly_positive(0.5);

        let mut c: StrictlyPositive = a * a;
        assert_close(c.get(), 0.01);
        c /= a;
        assert_close(c.get(), 0.1);
        let c: StrictlyPositive = a / b;
        assert_close(c.get(), 0.2);
        let mut c: StrictlyPositive = a + b;
        c += b;
        assert_close(c.get(), 1.1);

        let d: f64 = b - a;
        assert_close(d, 0.4);
        assert!(StrictlyPositive::new(b - b).is_err());
        let d: f64 = -b;
        assert_eq!(d, -0.5);
    }

    #[test]
    fn zero_one_open_arithmetic() {
        let a = zero_one_open(0.1);
        let b = zero_one_open(0.5);
        assert_eq!("0.99".parse::<ZeroOneOpen>().unwrap(), 0.99);

        let mut c: ZeroOneOpen = a * a;
        assert_close(c.get(), 0.01);
        c *= b;
        assert_close(c.get(), 0.005);

        let d: f64 = a / b;
        assert_close(d, 0.2);
        assert!(ZeroOneOpen::new(b / a).is_err());
        assert!(ZeroOneOpen::new(a / a).is_err());
        let d: f64 = a + b;
        assert_close(d, 0.6);
        assert!(ZeroOneOpen::new(a + zero_one_open(0.9)).is_err());
        let d: f64 = b - a;
        assert_close(d, 0.4);
        assert!(ZeroOneOpen::new(a - a).is_err());
        let d: f64 = -b;
        assert_eq!(d, -0.5);
    }

    #[test]
    fn unbounded_arithmetic() {
        let a = unbounded(0.1);
        let b = unbounded(0.5);
        assert!(Unbounded::new(-0.1).is_ok());
        assert_eq!("14.31".parse::<Unbounded>().unwrap(), 14.31);

        let mut c: Unbounded = a * a;
        c /= a;
        assert_close(c.get(), 0.1);
        let c: Unbounded = a / b;
        assert_close(c.get(), 0.2);
        let mut c: Unbounded = a + b;
        c += b;
        assert_close(c.get(), 1.1);
        let mut c: Unbounded = b - a;
        c -= a;
        assert_close(c.get(), 0.3);
        c -= b;
        assert_close(c.get(), -0.2);
        let d: Unbounded = -b;
        assert_eq!(d, -0.5);
    }

    #[test]
    fn negative_intervals_are_closed_under_addition_only() {
        let a = Negative::new(-0.1).unwrap();
        let b = Negative::new(-0.5).unwrap();
        let mut c: Negative = a + b;
        c += b;
        assert_close(c.get(), -1.1);
        let d: f64 = a - b;
        assert_close(d, 0.4);
        let d: f64 = a * b;
        assert_close(d, 0.05);
        let d: f64 = a / b;
        assert_close(d, 0.2);
        let d: f64 = -a;
        assert_eq!(d, 0.1);

        let a = StrictlyNegative::new(-0.1).unwrap();
        let b = StrictlyNegative::new(-0.5).unwrap();
        let mut c: StrictlyNegative = a + b;
        c += b;
        assert_close(c.get(), -1.1);
        let d: f64 = a - a;
        assert_eq!(d, 0.0);
        let d: f64 = a * b;
        assert_close(d, 0.05);
        let d: f64 = a / b;
        assert_close(d, 0.2);
        let d: f64 = -a;
        assert_eq!(d, 0.1);
    }

    #[test]
    fn unit_intervals_are_closed_under_multiplication_only() {
        let a = ZeroOneClosed::new(0.0).unwrap();
        let b = ZeroOneClosed::new(1.0).unwrap();
        let mut c: ZeroOneClosed = a * b;
        assert_eq!(c, 0.0);
        c *= b;
        assert_eq!(c, 0.0);
        let d: f64 = b + b;
        assert_eq!(d, 2.0);
        let d: f64 = a - b;
        assert_eq!(d, -1.0);
        let d: f64 = b / b;
        assert_eq!(d, 1.0);

        let a = ZeroOpenOneClosed::new(0.5).unwrap();
        let b = ZeroOpenOneClosed::new(1.0).unwrap();
        let mut c: ZeroOpenOneClosed = a * b;
        c *= a;
        assert_eq!(c, 0.25);
        let d: f64 = b + b;
        assert_eq!(d, 2.0);
        let d: f64 = a - a;
        assert_eq!(d, 0.0);
        let d: f64 = b / a;
        assert_eq!(d, 2.0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "0 is outside the interval (0, 1)")]
    fn closed_arithmetic_leaving_the_interval_numerically_is_a_dev_error() {
        let tiny = zero_one_open(1e-200);
        let _ = tiny * tiny; // underflows to 0
    }

    fn uint8(v: u8) -> UnsignedInt8 {
        UnsignedInt8::new(v).unwrap()
    }

    #[test]
    fn unsigned_int_arithmetic() {
        let a = uint8(2);
        let b = uint8(5);
        assert_eq!("14".parse::<UnsignedInt8>().unwrap(), 14);
        assert!(a < b);
        assert!(a < 10);
        assert!(a != 10);

        let mut c: UnsignedInt8 = a * a;
        assert_eq!(c, 4);
        c /= a;
        assert_eq!(c, 2);
        let c: UnsignedInt8 = a / b; // integer division
        assert_eq!(c, 0);

        let mut c: UnsignedInt8 = a + b;
        assert_eq!(c, 7);
        c += b;
        assert_eq!(c, 12);
        c += uint8(243);
        assert_eq!(c, 255);
        c *= uint8(1);
        assert_eq!(c, 255);

        let mut d: UnsignedInt8 = b - a;
        assert_eq!(d, 3);
        d -= uint8(3);
        assert_eq!(d, 0);
    }

    #[test]
    #[should_panic(expected = "0 - 1 is outside the range of u8")]
    fn unsigned_compound_subtraction_below_zero_is_a_dev_error() {
        let mut c = uint8(0);
        c -= uint8(1);
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn unsigned_division_by_zero_is_a_dev_error() {
        let _ = uint8(1) / uint8(0);
    }

    #[test]
    #[should_panic(expected = "is outside the range of usize")]
    fn strictly_positive_uint_overflow_is_a_dev_error() {
        let one = StrictlyPositiveUInt::new(1).unwrap();
        let _ = StrictlyPositiveUInt::MAX + one;
    }

    #[test]
    #[should_panic(expected = "100 + 156 is outside the range of u8")]
    fn unsigned_addition_overflow_is_a_dev_error() {
        let _ = uint8(100) + uint8(156);
    }

    #[test]
    #[should_panic(expected = "255 + 1 is outside the range of u8")]
    fn unsigned_compound_addition_overflow_is_a_dev_error() {
        let mut c = uint8(255);
        c += uint8(1);
    }

    #[test]
    #[should_panic(expected = "100 * 5 is outside the range of u8")]
    fn unsigned_multiplication_overflow_is_a_dev_error() {
        let _ = uint8(100) * uint8(5);
    }

    #[test]
    #[should_panic(expected = "2 - 5 is outside the range of u8")]
    fn unsigned_subtraction_below_zero_is_a_dev_error() {
        let _ = uint8(2) - uint8(5);
    }

    #[test]
    fn unsigned_multiplication_by_zero_is_zero() {
        assert_eq!(uint8(0) * uint8(7), 0);
        assert_eq!(uint8(7) * uint8(0), 0);
        let mut c = UnsignedInt64::new(0).unwrap();
        c *= UnsignedInt64::MAX;
        assert_eq!(c, 0);
    }

    #[test]
    fn strictly_positive_uint_division_returns_the_raw_number() {
        let one = StrictlyPositiveUInt::new(1).unwrap();
        let two = StrictlyPositiveUInt::new(2).unwrap();
        let q: usize = one / two;
        assert_eq!(q, 0);
        let mut c: StrictlyPositiveUInt = one + two;
        c *= two;
        assert_eq!(c, 6);
        let d: usize = one - one;
        assert_eq!(d, 0);
    }
}
