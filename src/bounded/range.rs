//! Numeric ranges with an included or excluded end on each side, in the C++ coretools range syntax.

use core::fmt;
use core::ops::{Bound, RangeBounds};
use core::str::FromStr;

use num_traits::ToPrimitive;

use super::BoundedError;

/// A range of numbers whose ends are each included or excluded, e.g. `[0, 1)`.
///
/// An unbounded end is the type's extreme value, included: the default range spans every value of `T`.
/// The range may be empty (`(5, 5)`), but its minimum is never above its maximum, and its ends are
/// never infinite or NaN. It implements [`RangeBounds`], e.g. for `BTreeMap::range`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericRange<T> {
    min: T,
    max: T,
    min_included: bool,
    max_included: bool,
}

impl<T: num_traits::Bounded> Default for NumericRange<T> {
    fn default() -> Self {
        Self {
            min: T::min_value(),
            max: T::max_value(),
            min_included: true,
            max_included: true,
        }
    }
}

/// The value and inclusion of one end; an unbounded end is `extreme`, included.
fn split<T>(bound: Bound<T>, extreme: T) -> (T, bool) {
    match bound {
        Bound::Included(v) => (v, true),
        Bound::Excluded(v) => (v, false),
        Bound::Unbounded => (extreme, true),
    }
}

impl<T: Copy + PartialOrd + fmt::Display + num_traits::Bounded> NumericRange<T> {
    /// Range from `min` to `max`; an unbounded end becomes the type's extreme value, included.
    pub fn new(min: Bound<T>, max: Bound<T>) -> Result<Self, BoundedError> {
        let (min, min_included) = split(min, T::min_value());
        let (max, max_included) = split(max, T::max_value());
        // Only float ends can lie outside the type's extremes: infinities and NaN.
        for end in [min, max] {
            if !(T::min_value()..=T::max_value()).contains(&end) {
                return Err(BoundedError::OutOfInterval {
                    interval: "(-inf, inf)",
                    value: end.to_string(),
                });
            }
        }
        if min > max {
            return Err(BoundedError::MinAboveMax {
                min: min.to_string(),
                max: max.to_string(),
            });
        }
        Ok(Self {
            min,
            max,
            min_included,
            max_included,
        })
    }

    pub fn min(&self) -> T {
        self.min
    }

    pub fn max(&self) -> T {
        self.max
    }

    pub fn min_is_included(&self) -> bool {
        self.min_included
    }

    pub fn max_is_included(&self) -> bool {
        self.max_included
    }

    /// Whether `value` lies in the range (`false` for NaN).
    pub fn contains(&self, value: T) -> bool {
        let above_min = if self.min_included {
            value >= self.min
        } else {
            value > self.min
        };
        let below_max = if self.max_included {
            value <= self.max
        } else {
            value < self.max
        };
        above_min && below_max
    }

    /// Whether `value` lies below the range (`false` for NaN).
    pub fn is_below(&self, value: T) -> bool {
        if self.min_included {
            value < self.min
        } else {
            value <= self.min
        }
    }

    /// Whether `value` lies above the range (`false` for NaN).
    pub fn is_above(&self, value: T) -> bool {
        if self.max_included {
            value > self.max
        } else {
            value >= self.max
        }
    }
}

impl<T> RangeBounds<T> for NumericRange<T> {
    fn start_bound(&self) -> Bound<&T> {
        if self.min_included {
            Bound::Included(&self.min)
        } else {
            Bound::Excluded(&self.min)
        }
    }

    fn end_bound(&self) -> Bound<&T> {
        if self.max_included {
            Bound::Included(&self.max)
        } else {
            Bound::Excluded(&self.max)
        }
    }
}

/// Parses the C++ coretools range syntax: `[a,b]`, `(a,b]`, `]a,b[`, `a,b`, …
///
/// `(` or `]` before the minimum and `)` or `[` after the maximum exclude that end; `[`, `]` or no
/// bracket include it. An empty bound, `-inf` as minimum or `inf` as maximum is the type's extreme.
/// Whitespace around the range and the bounds is ignored.
impl<T> FromStr for NumericRange<T>
where
    T: Copy + PartialOrd + fmt::Display + num_traits::Bounded + FromStr<Err: Into<BoundedError>>,
{
    type Err = BoundedError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let invalid = |reason| BoundedError::InvalidRange {
            input: input.to_owned(),
            reason,
        };
        let range = input.trim();
        if range == "," {
            return Err(invalid("a lone ',' is not a range"));
        }

        let (min_included, range) = match range.as_bytes().first() {
            Some(b'(' | b']') => (false, &range[1..]),
            Some(b'[') => (true, &range[1..]),
            _ => (true, range),
        };
        let (min, max) = range
            .split_once(',')
            .ok_or_else(|| invalid("no ',' between the bounds"))?;
        let (max_included, max) = match max.as_bytes().last() {
            Some(b')' | b'[') => (false, &max[..max.len() - 1]),
            Some(b']') => (true, &max[..max.len() - 1]),
            _ => (true, max),
        };

        // An omitted bound is the type's extreme, still included or excluded by its bracket.
        let parse = |bound: &str, infinite: &str, extreme: T, included: bool| {
            let bound = bound.trim();
            let value = if bound.is_empty() || bound == infinite {
                extreme
            } else {
                bound
                    .parse()
                    .map_err(|e: T::Err| BoundedError::InvalidBound {
                        input: input.to_owned(),
                        source: Box::new(e.into()),
                    })?
            };
            Ok::<_, BoundedError>(if included {
                Bound::Included(value)
            } else {
                Bound::Excluded(value)
            })
        };
        Self::new(
            parse(min, "-inf", T::min_value(), min_included)?,
            parse(max, "inf", T::max_value(), max_included)?,
        )
    }
}

/// Prints the range in the syntax [`FromStr`] reads, e.g. `[0,1)`. An end at the type's extreme prints
/// as `-inf` / `inf` when that extreme is at least 1000 in magnitude (`[-inf,inf]` for `f64`,
/// `[0,inf]` for `u16`, but `[-128,127]` for `i8`).
impl<T> fmt::Display for NumericRange<T>
where
    T: fmt::Display + PartialEq + num_traits::Bounded + ToPrimitive,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.min_included { "[" } else { "(" })?;
        let lowest = T::min_value();
        if self.min == lowest && lowest.to_f64().is_some_and(|v| v <= -1e3) {
            f.write_str("-inf")?;
        } else {
            write!(f, "{}", self.min)?;
        }
        f.write_str(",")?;
        let highest = T::max_value();
        if self.max == highest && highest.to_f64().is_some_and(|v| v >= 1e3) {
            f.write_str("inf")?;
        } else {
            write!(f, "{}", self.max)?;
        }
        f.write_str(if self.max_included { "]" } else { ")" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParseNumberError;
    use crate::bounded::ZeroOneClosed;
    use core::ops::Bound::{Excluded, Included, Unbounded};

    fn zero_one(v: f64) -> ZeroOneClosed {
        ZeroOneClosed::new(v).unwrap()
    }

    // --- ported from C++ TNumericRangeUnitTests.cpp ---------------------------
    // `TNumericRange<Probability>` is ported as `NumericRange<ZeroOneClosed>`, the
    // bounded value with the same interval. Not ported: `constuctfromParameters`
    // (TParameters is replaced by clap, which parses through `FromStr`).

    #[test]
    fn default_range_spans_the_type() {
        let u8 = NumericRange::<u8>::default();
        assert!(u8.contains(4));
        assert!(u8.contains(u8::MAX));
        assert!(u8.contains(u8::MIN));

        let d = NumericRange::<f64>::default();
        assert!(d.contains(4.0));
        assert!(d.contains(f64::MAX));
        assert!(d.contains(f64::MIN));
    }

    #[test]
    fn closed_range() {
        let u8 = NumericRange::new(Included(10u8), Included(100)).unwrap();
        for v in [10, 40, 100] {
            assert!(u8.contains(v), "{v}");
        }
        for v in [4, 104] {
            assert!(!u8.contains(v), "{v}");
        }

        let d = NumericRange::new(Included(-9.99), Included(100.01)).unwrap();
        for v in [-9.99, -4.0, 40.0, 100.01] {
            assert!(d.contains(v), "{v}");
        }
        for v in [-9.991, 100.011] {
            assert!(!d.contains(v), "{v}");
        }
    }

    #[test]
    fn range_excluding_max() {
        let u8 = NumericRange::new(Included(10u8), Excluded(100)).unwrap();
        assert!(u8.contains(10));
        assert!(u8.contains(40));
        for v in [9, 100, 101] {
            assert!(!u8.contains(v), "{v}");
        }
        assert!(u8.is_above(100));
        assert!(u8.is_below(9));
        assert!(!u8.is_above(9));
        assert!(!u8.is_below(10));

        let d = NumericRange::new(Included(-9.99), Excluded(100.01)).unwrap();
        for v in [-9.99, -4.0, 40.0] {
            assert!(d.contains(v), "{v}");
        }
        for v in [-9.991, 100.01, 100.011] {
            assert!(!d.contains(v), "{v}");
        }
        assert!(d.is_above(100.01));
        assert!(d.is_below(-10.0));
        assert!(!d.is_above(100.0));
        assert!(!d.is_below(-9.99));
    }

    #[test]
    fn range_excluding_min() {
        let u8 = NumericRange::new(Excluded(10u8), Included(100)).unwrap();
        assert!(u8.contains(40));
        assert!(u8.contains(100));
        for v in [4, 10, 104] {
            assert!(!u8.contains(v), "{v}");
        }

        let d = NumericRange::new(Excluded(-9.99), Included(100.01)).unwrap();
        for v in [-4.0, 40.0, 100.01] {
            assert!(d.contains(v), "{v}");
        }
        for v in [-9.991, -9.99, 100.011] {
            assert!(!d.contains(v), "{v}");
        }
        assert!(d.is_above(100.02));
        assert!(d.is_below(-9.99));
        assert!(!d.is_above(100.01));
        assert!(!d.is_below(-9.989));
    }

    #[test]
    fn open_range() {
        let u8 = NumericRange::new(Excluded(10u8), Excluded(100)).unwrap();
        assert!(u8.contains(40));
        for v in [4, 10, 100, 104] {
            assert!(!u8.contains(v), "{v}");
        }

        let d = NumericRange::new(Excluded(-9.99), Excluded(100.01)).unwrap();
        assert!(d.contains(-4.0));
        assert!(d.contains(40.0));
        for v in [-9.991, -9.99, 100.01, 100.011] {
            assert!(!d.contains(v), "{v}");
        }
    }

    #[test]
    fn parse_closed_range() {
        let u8: NumericRange<u8> = "[10,100]".parse().unwrap();
        assert!(u8.min_is_included());
        assert!(u8.max_is_included());
        assert_eq!(u8.min(), 10);
        assert_eq!(u8.max(), 100);
        for v in [10, 40, 100] {
            assert!(u8.contains(v), "{v}");
        }
        for v in [9, 101] {
            assert!(!u8.contains(v), "{v}");
        }

        let d: NumericRange<f64> = "[-9.99,100.01]".parse().unwrap();
        assert!(d.min_is_included());
        assert!(d.max_is_included());
        assert_eq!(d.min(), -9.99);
        assert_eq!(d.max(), 100.01);
        for v in [-9.99, -4.0, 40.0, 100.01] {
            assert!(d.contains(v), "{v}");
        }
        for v in [-9.991, 100.011] {
            assert!(!d.contains(v), "{v}");
        }
    }

    #[test]
    fn parse_half_open_ranges() {
        let u8: NumericRange<u8> = "[10,100)".parse().unwrap();
        assert!(u8.min_is_included());
        assert!(!u8.max_is_included());
        assert!(u8.contains(10));
        assert!(!u8.contains(100));
        let d: NumericRange<f64> = "[-9.99,100.01)".parse().unwrap();
        assert!(d.contains(-9.99));
        assert!(!d.contains(100.01));

        let u8: NumericRange<u8> = "(10 ,100]".parse().unwrap();
        assert!(!u8.min_is_included());
        assert!(u8.max_is_included());
        assert!(!u8.contains(10));
        assert!(u8.contains(100));
        let d: NumericRange<f64> = "(-9.99, 100.01]".parse().unwrap();
        assert!(!d.contains(-9.99));
        assert!(d.contains(100.01));

        let d: NumericRange<f64> = "[0.2, 0.5[".parse().unwrap();
        assert!(d.min_is_included());
        assert!(!d.max_is_included());
        let d: NumericRange<f64> = "]0.2, 0.5]".parse().unwrap();
        assert!(!d.min_is_included());
        assert!(d.max_is_included());
    }

    #[test]
    fn parse_open_range() {
        let u8: NumericRange<u8> = " (10,100)".parse().unwrap();
        assert!(!u8.min_is_included());
        assert!(!u8.max_is_included());
        assert!(u8.contains(40));
        for v in [4, 10, 100, 104] {
            assert!(!u8.contains(v), "{v}");
        }

        let d: NumericRange<f64> = "(-9.99,100.01) ".parse().unwrap();
        assert!(d.contains(-4.0));
        for v in [-9.991, -9.99, 100.01, 100.011] {
            assert!(!d.contains(v), "{v}");
        }
        let d: NumericRange<f64> = "]-9.99,100.01[".parse().unwrap();
        assert_eq!(d, "(-9.99,100.01)".parse().unwrap());
    }

    #[test]
    fn parse_omitted_brackets_and_bounds() {
        let u8: NumericRange<u8> = "10,100)".parse().unwrap();
        assert!(u8.min_is_included());
        assert!(!u8.max_is_included());

        let u16: NumericRange<u16> = "(10,".parse().unwrap();
        assert!(!u16.min_is_included());
        assert!(u16.max_is_included());
        assert!(u16.contains(65335));

        let p: NumericRange<ZeroOneClosed> = "0.2,inf".parse().unwrap();
        assert!(p.min_is_included());
        assert!(p.max_is_included());
        assert_eq!(p.min(), 0.2);
        assert_eq!(p.max(), 1.0);

        let s: NumericRange<usize> = "-inf,123)".parse().unwrap();
        assert!(s.min_is_included());
        assert!(!s.max_is_included());
        assert_eq!(s.min(), 0);
        assert_eq!(s.max(), 123);

        let d: NumericRange<f64> = " [ , 100.01) ".parse().unwrap();
        assert!(d.min_is_included());
        assert!(!d.max_is_included());
        assert!(d.contains(-10000.0));
        assert!(!d.contains(100.01));

        let u8: NumericRange<u8> = "(,5]".parse().unwrap();
        assert_eq!(u8.min(), 0);
        assert!(!u8.min_is_included());
        assert!(!u8.contains(0));
    }

    #[test]
    fn parse_rejects_malformed_ranges() {
        for s in [
            "10;100",
            "4,10,100",
            "10,100X",
            "(10;100)",
            "(10F0,10R)",
            ",",
            "(20,10)",
        ] {
            assert!(s.parse::<NumericRange<f64>>().is_err(), "{s}");
        }
        let err = "10;100".parse::<NumericRange<f64>>().unwrap_err();
        assert!(matches!(err, BoundedError::InvalidRange { .. }));
        let err = "(10F0,10R)".parse::<NumericRange<f64>>().unwrap_err();
        let BoundedError::InvalidBound { source, .. } = err else {
            panic!("{err:?}");
        };
        assert!(matches!(
            *source,
            BoundedError::NotANumber(ParseNumberError::Float(_))
        ));
    }

    #[test]
    fn ends_beyond_the_type_extremes_are_rejected() {
        for end in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                matches!(
                    NumericRange::new(Included(end), Unbounded),
                    Err(BoundedError::OutOfInterval { .. })
                ),
                "{end}"
            );
            assert!(
                NumericRange::new(Unbounded, Excluded(end)).is_err(),
                "{end}"
            );
        }
        for s in ["(0,infinity)", "(-1e400,0)", "[NaN,1]"] {
            assert!(s.parse::<NumericRange<f64>>().is_err(), "{s}");
        }
    }

    #[test]
    fn range_of_bounded_values_is_limited_to_their_interval() {
        let p = NumericRange::<ZeroOneClosed>::default();
        assert_eq!(p.min(), 0.0);
        assert_eq!(p.max(), 1.0);

        let p: NumericRange<ZeroOneClosed> = "[0.2, 0.5[".parse().unwrap();
        assert_eq!(p.min(), 0.2);
        assert!(p.min_is_included());
        assert_eq!(p.max(), 0.5);
        assert!(!p.max_is_included());

        for s in ["-2,", ",1.3"] {
            let err = s.parse::<NumericRange<ZeroOneClosed>>().unwrap_err();
            let BoundedError::InvalidBound { source, .. } = err else {
                panic!("{s}: {err:?}");
            };
            assert!(
                matches!(*source, BoundedError::OutOfInterval { .. }),
                "{s}: {source}"
            );
        }
    }

    #[test]
    #[expect(clippy::approx_constant, reason = "bounds of the C++ test")]
    fn display_in_range_syntax() {
        let p = NumericRange::<ZeroOneClosed>::default();
        assert_eq!(p.to_string(), "[0,1]");
        let p = NumericRange::new(Excluded(zero_one(0.1)), Included(zero_one(0.2))).unwrap();
        assert_eq!(p.to_string(), "(0.1,0.2]");

        assert_eq!(NumericRange::<usize>::default().to_string(), "[0,inf]");
        let s = NumericRange::new(Included(3usize), Excluded(333)).unwrap();
        assert_eq!(s.to_string(), "[3,333)");

        assert_eq!(NumericRange::<f64>::default().to_string(), "[-inf,inf]");
        let d = NumericRange::new(Excluded(-3.14), Excluded(2.72)).unwrap();
        assert_eq!(d.to_string(), "(-3.14,2.72)");

        assert_eq!(NumericRange::<i8>::default().to_string(), "[-128,127]");
        assert_eq!(NumericRange::<u16>::default().to_string(), "[0,inf]");
    }

    // --- divergences from C++ -------------------------------------------------

    #[test]
    fn parse_rejects_an_empty_string() {
        for s in ["", "   "] {
            let err = s.parse::<NumericRange<f64>>().unwrap_err();
            assert!(matches!(err, BoundedError::InvalidRange { .. }), "{s:?}");
        }
    }

    #[test]
    fn display_round_trips_every_bound() {
        for s in [
            "[0.1234567,0.12345678)",
            "(-1e-7,1e300]",
            "[-inf,123456789.5)",
        ] {
            let range: NumericRange<f64> = s.parse().unwrap();
            assert_eq!(
                range.to_string().parse::<NumericRange<f64>>().unwrap(),
                range,
                "{s}"
            );
        }
        let range: NumericRange<f64> = "[0.1234567,0.12345678)".parse().unwrap();
        assert_eq!(range.to_string(), "[0.1234567,0.12345678)");
    }

    #[test]
    fn nan_lies_neither_within_nor_outside_a_range() {
        let d = NumericRange::<f64>::default();
        assert!(!d.contains(f64::NAN));
        assert!(!d.is_below(f64::NAN));
        assert!(!d.is_above(f64::NAN));
    }

    // --- new behaviour -------------------------------------------------------

    #[test]
    fn unbounded_end_is_the_included_type_extreme() {
        let s = NumericRange::new(Unbounded, Excluded(123usize)).unwrap();
        assert_eq!(s.min(), 0);
        assert!(s.min_is_included());
        assert_eq!(s.max(), 123);
        assert!(!s.max_is_included());

        let d = NumericRange::new(Included(0.0), Unbounded).unwrap();
        assert_eq!(d.max(), f64::MAX);
        assert!(d.max_is_included());
    }

    #[test]
    fn min_above_max_is_an_error() {
        let err = NumericRange::new(Included(5u8), Included(4)).unwrap_err();
        assert_eq!(err.to_string(), "range minimum 5 is above its maximum 4");
        assert!(NumericRange::new(Included(f64::NAN), Unbounded).is_err());
        assert!(NumericRange::new(Excluded(5u8), Excluded(5)).is_ok());
    }
}
