//! Interval markers: zero-sized types that name the interval of a [`Bounded`](super::Bounded) value.
//!
//! Every marker is defined for `f64`; [`Positive`] and [`StrictlyPositive`] are also defined for the
//! unsigned integers. For `f64`, an interval with an open end excludes exactly the end point: the
//! smallest [`StrictlyPositive`] value is the smallest positive subnormal, the largest [`ZeroOneOpen`]
//! value is the largest `f64` below 1. Infinities and NaN lie outside every interval.

use super::Interval;

macro_rules! marker {
    ($($(#[$m:meta])* $name:ident;)*) => {$(
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $name;
    )*};
}

marker! {
    /// Every finite number: `(-inf, inf)`.
    Unbounded;
    /// `[0, 1]`.
    ZeroOneClosed;
    /// `(0, 1)`.
    ZeroOneOpen;
    /// `(0, 1]`.
    ZeroOpenOneClosed;
    /// `[0, inf)`.
    Positive;
    /// `(0, inf)`.
    StrictlyPositive;
    /// `(-inf, 0]`.
    Negative;
    /// `(-inf, 0)`.
    StrictlyNegative;
}

macro_rules! bounds {
    ($t:ty: $($marker:ident [$min:expr, $max:expr] $notation:expr),* $(,)?) => {$(
        impl Interval<$t> for $marker {
            const MIN: $t = $min;
            const MAX: $t = $max;
            const NOTATION: &'static str = $notation;
        }
    )*};
}

/// Smallest positive `f64` (a subnormal).
const F64_TINY: f64 = f64::from_bits(1);
/// Largest `f64` below 1.
const F64_BELOW_ONE: f64 = 1.0 - f64::EPSILON / 2.0;

bounds! { f64:
    Unbounded [f64::MIN, f64::MAX] "(-inf, inf)",
    ZeroOneClosed [0.0, 1.0] "[0, 1]",
    ZeroOneOpen [F64_TINY, F64_BELOW_ONE] "(0, 1)",
    ZeroOpenOneClosed [F64_TINY, 1.0] "(0, 1]",
    Positive [0.0, f64::MAX] "[0, inf)",
    StrictlyPositive [F64_TINY, f64::MAX] "(0, inf)",
    Negative [f64::MIN, 0.0] "(-inf, 0]",
    StrictlyNegative [f64::MIN, -F64_TINY] "(-inf, 0)",
}

// `$max` is `<$t>::MAX` spelled out, so that the notation is a `&'static str`.
macro_rules! unsigned_bounds {
    ($($t:ty: $max:literal),*) => {$(
        bounds! { $t:
            Positive [0, <$t>::MAX] concat!("[0, ", $max, "]"),
            StrictlyPositive [1, <$t>::MAX] concat!("[1, ", $max, "]"),
        }
    )*};
}

unsigned_bounds!(u8: "255", u16: "65535", u32: "4294967295", u64: "18446744073709551615");
#[cfg(target_pointer_width = "64")]
unsigned_bounds!(usize: "18446744073709551615");
#[cfg(target_pointer_width = "32")]
unsigned_bounds!(usize: "4294967295");
#[cfg(target_pointer_width = "16")]
unsigned_bounds!(usize: "65535");
