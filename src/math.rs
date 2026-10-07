mod math_constants;

use core::cmp::Ordering;
use core::error::Error;
use core::f64::consts::{LN_10, LOG10_E};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};
use core::str::FromStr;

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use num_traits::{NumAssign, One, Zero};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ProbabilityError {
    #[error("{value} is outside of the {kind} probability range")]
    OutOfRange { kind: &'static str, value: String },

    #[error("cannot parse {input:?} as a {kind} probability")]
    Parse {
        kind: &'static str,
        input: String,
        #[source]
        source: Box<dyn Error + Send + Sync + 'static>,
    },
}

/// Compile-time description of a probability representation.
/// Equivalent to `TConverter<Type>` plus the `static constexpr` queries of the class.
pub trait ProbabilityKind: Copy + PartialEq + PartialOrd + fmt::Debug + 'static {
    type Value: Copy
        + PartialOrd
        + NumAssign
        + num_traits::NumCast
        + fmt::Debug
        + fmt::Display
        + FromStr<Err: Error + Send + Sync + 'static>;

    const NAME: &'static str;
    const MIN: Self::Value;
    const MAX: Self::Value;
    const IS_LINEAR: bool;
    const IS_PHREDED: bool;

    #[inline]
    fn is_valid(v: Self::Value) -> bool {
        if Self::IS_LINEAR {
            v >= Self::MIN && v <= Self::MAX
        } else {
            v <= Self::MAX // also rejects NaN for float kinds
        }
    }
}

macro_rules! kind {
    ($(#[$m:meta])* $ty:ident, $name:literal, $v:ty, $min:expr, $max:expr, linear: $lin:literal, phreded: $ph:literal) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $ty;
        impl ProbabilityKind for $ty {
            type Value = $v;
            const NAME: &'static str = $name;
            const MIN: $v = $min;
            const MAX: $v = $max;
            const IS_LINEAR: bool = $lin;
            const IS_PHREDED: bool = $ph;
        }
    };
}

kind!(/** Plain probability in `[0, 1]`. */ Linear, "linear", f64, 0.0, 1.0, linear: true, phreded: false);
kind!(/** Natural log, `(-inf, 0]`. */ Log, "log", f64, f64::MIN, 0.0, linear: false, phreded: false);
kind!(/** Base-10 log, `(-inf, 0]`. */ Log10, "log10", f64, f64::MIN, 0.0, linear: false, phreded: false);
kind!(/** `-10 * log10(p)`, rounded. */ Phred, "phred", u8, 0, u8::MAX, linear: false, phreded: true);
kind!(/** `-1000 * log10(p)`, rounded; `u16::MAX` is reserved as tag. */
      HpPhred, "hpPhred", u16, 0, u16::MAX - 1, linear: false, phreded: true);

/// Kinds that support `+=` (every kind except [`Linear`]).
pub trait NonLinear: ProbabilityKind {}
impl NonLinear for Log {}
impl NonLinear for Log10 {}
impl NonLinear for Phred {}
impl NonLinear for HpPhred {}

/// Kinds that can hold a sentinel "tag" value (`isTaggable()`: all but [`Phred`]).
pub trait Taggable: ProbabilityKind {
    const TAG: Self::Value;
    fn is_tag(v: Self::Value) -> bool;
}
/// Kinds that can encode an arbitrary byte as tag (`!isPhreded()`).
pub trait ByteTaggable: Taggable {
    fn encode_tag(tag: u8) -> Self::Value;
    fn decode_tag(v: Self::Value) -> Option<u8>;
}

impl Taggable for Linear {
    const TAG: f64 = -1.0; // setAsTag(0)
    fn is_tag(v: f64) -> bool {
        v < 0.0
    }
}
impl ByteTaggable for Linear {
    fn encode_tag(tag: u8) -> f64 {
        -(f64::from(tag) + 1.0)
    }
    fn decode_tag(v: f64) -> Option<u8> {
        let t = -v - 1.0;
        (0.0..=255.0).contains(&t).then_some(t as u8)
    }
}

macro_rules! log_tag {
    ($ty:ident) => {
        impl Taggable for $ty {
            const TAG: f64 = 1.0; // setAsTag(0)
            fn is_tag(v: f64) -> bool {
                v > 0.0
            }
        }
        impl ByteTaggable for $ty {
            fn encode_tag(tag: u8) -> f64 {
                f64::from(tag) + 1.0
            }
            fn decode_tag(v: f64) -> Option<u8> {
                let t = v - 1.0;
                (0.0..=255.0).contains(&t).then_some(t as u8)
            }
        }
    };
}
log_tag!(Log);
log_tag!(Log10);

impl Taggable for HpPhred {
    const TAG: u16 = u16::MAX; // max() + 1
    fn is_tag(v: u16) -> bool {
        v == Self::TAG
    }
}

// ---------------------------------------------------------------------------
// Conversions (replaces `TConverter<To>::from<From>(t)`)
// ---------------------------------------------------------------------------

/// `To: ConvertFrom<From>` means a `Probability<From>` can become a `Probability<To>`.
pub trait ConvertFrom<F: ProbabilityKind>: ProbabilityKind {
    fn convert(v: F::Value) -> Self::Value;
}

impl<K: ProbabilityKind> ConvertFrom<K> for K {
    #[inline]
    fn convert(v: K::Value) -> K::Value {
        v
    }
}

/// Round a (high-precision) phred score into an integer type, saturating at `max`.
/// Negative values and NaN (which were UB in the C++ cast) map to 0.
#[inline]
fn quantize<T: num_traits::NumCast + Zero + Copy>(ph: f64, max: T) -> T {
    let max_f = max.to_f64().expect("integer max fits in f64");
    if ph > max_f {
        return max;
    }
    T::from(ph.round()).unwrap_or_else(T::zero)
}

macro_rules! convert {
    ($($from:ident => $to:ident : |$t:ident| $body:expr;)*) => {$(
        impl ConvertFrom<$from> for $to {
            #[inline]
            fn convert($t: <$from as ProbabilityKind>::Value) -> <$to as ProbabilityKind>::Value {
                $body
            }
        }
        impl From<Probability<$from>> for Probability<$to> {
            #[inline]
            fn from(p: Probability<$from>) -> Self {
                p.convert()
            }
        }
    )*};
}

convert! {
    // -> linear  (C++ uses PhredTables lookups for phred/hpPhred; swap in a LazyLock table if hot)
    Log     => Linear: |t| t.exp();
    Log10   => Linear: |t| 10f64.powf(t);
    Phred   => Linear: |t| 10f64.powf(f64::from(t) / -10.0);
    HpPhred => Linear: |t| 10f64.powf(f64::from(t) / -1000.0);
    // -> log
    Linear  => Log: |t| t.ln();
    Log10   => Log: |t| LN_10 * t;
    Phred   => Log: |t| -0.1 * LN_10 * f64::from(t);
    HpPhred => Log: |t| -0.001 * LN_10 * f64::from(t);
    // -> log10
    Linear  => Log10: |t| t.log10();
    Log     => Log10: |t| LOG10_E * t;
    Phred   => Log10: |t| -0.1 * f64::from(t);
    HpPhred => Log10: |t| -0.001 * f64::from(t);
    // -> phred
    Linear  => Phred: |t| quantize(-10.0 * t.log10(), Phred::MAX);
    Log     => Phred: |t| quantize(-10.0 * LOG10_E * t, Phred::MAX);
    Log10   => Phred: |t| quantize(-10.0 * t, Phred::MAX);
    HpPhred => Phred: |t| quantize(f64::from(t) / 100.0, Phred::MAX);
    // -> hpPhred
    Linear  => HpPhred: |t| quantize(-1000.0 * t.log10(), HpPhred::MAX);
    Log     => HpPhred: |t| quantize(-1000.0 * LOG10_E * t, HpPhred::MAX);
    Log10   => HpPhred: |t| quantize(-1000.0 * t, HpPhred::MAX);
    Phred   => HpPhred: |t| u16::from(t) * 100;
}

// ---------------------------------------------------------------------------
// Probability<K>  (replaces TSomeProbability<Type>)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability<K: ProbabilityKind> {
    value: K::Value,
    _kind: PhantomData<K>,
}

pub type LinearProbability = Probability<Linear>;
pub type LogProbability = Probability<Log>;
pub type Log10Probability = Probability<Log10>;
pub type PhredProbability = Probability<Phred>;
pub type HpPhredProbability = Probability<HpPhred>;

impl<K: ProbabilityKind> Default for Probability<K> {
    fn default() -> Self {
        Self::new_unchecked(K::MIN)
    }
}

impl<K: ProbabilityKind> Probability<K> {
    /// Checked constructor.
    pub fn new(value: K::Value) -> Result<Self, ProbabilityError> {
        if K::is_valid(value) {
            Ok(Self::new_unchecked(value))
        } else {
            Err(ProbabilityError::OutOfRange {
                kind: K::NAME,
                value: value.to_string(),
            })
        }
    }

    /// `TSomeProbability(tags::NoCheck, value)`. Not `unsafe`: an invalid value
    /// is a logic error, not undefined behaviour.
    #[inline]
    pub const fn new_unchecked(value: K::Value) -> Self {
        Self {
            value,
            _kind: PhantomData,
        }
    }

    #[inline]
    pub const fn get(self) -> K::Value {
        self.value
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        K::is_valid(self.value)
    }

    #[inline]
    fn debug_check(self) {
        debug_assert!(
            self.is_valid(),
            "{} probability cannot have value {}!",
            K::NAME,
            self.value
        );
    }

    /// Generic counterpart of the converting constructor, for use in generic code.
    /// For concrete kinds, `Probability::<Log>::from(p)` / `p.into()` also work.
    #[inline]
    pub fn convert<T: ConvertFrom<K>>(self) -> Probability<T> {
        Probability::new_unchecked(T::convert(self.value))
    }

    /// Divide the probability by `factor` (in this kind's representation).
    pub fn scale(&mut self, factor: K::Value) -> &mut Self {
        if K::IS_LINEAR {
            self.value /= factor;
        } else {
            self.value -= factor;
        }
        self.debug_check();
        self
    }

    pub fn more_probable(self, other: Self) -> bool {
        if K::IS_PHREDED {
            self.value < other.value
        } else {
            self.value > other.value
        }
    }

    pub const fn is_linear() -> bool {
        K::IS_LINEAR
    }
    pub const fn is_phreded() -> bool {
        K::IS_PHREDED
    }
    pub const fn min() -> K::Value {
        K::MIN
    }
    pub const fn max() -> K::Value {
        K::MAX
    }

    /// Least probable representable value.
    pub const fn lowest() -> Self {
        Self::new_unchecked(if K::IS_PHREDED { K::MAX } else { K::MIN })
    }

    /// Most probable representable value.
    pub const fn highest() -> Self {
        Self::new_unchecked(if K::IS_PHREDED { K::MIN } else { K::MAX })
    }
}

// --- linear-only -----------------------------------------------------------

impl Probability<Linear> {
    pub fn odds_ratio(self) -> f64 {
        self.value / (1.0 - self.value)
    }

    pub fn complement(self) -> Self {
        Self::new_unchecked(f64::one() - self.value)
    }
}

impl MulAssign for Probability<Linear> {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

// --- non-linear-only -------------------------------------------------------

impl<K: NonLinear> AddAssign for Probability<K> {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

// --- tags --------------------------------------------------------------------

impl<K: Taggable> Probability<K> {
    pub fn set_as_tag(&mut self) {
        self.value = K::TAG;
    }
    pub fn is_tag(self) -> bool {
        K::is_tag(self.value)
    }
}

impl<K: ByteTaggable> Probability<K> {
    pub fn set_as_byte_tag(&mut self, tag: u8) {
        self.value = K::encode_tag(tag);
    }
    /// `getAsTag()`: `None` if the stored value is not a tag
    /// (the C++ version only asserted this in checked builds).
    pub fn byte_tag(self) -> Option<u8> {
        K::decode_tag(self.value)
    }
}

// --- arithmetic: only between identical kinds ------------------------------

// `+`: linear -> raw value, others -> probability
impl Add for Probability<Linear> {
    type Output = f64;
    fn add(self, rhs: Self) -> f64 {
        self.value + rhs.value
    }
}
impl<K: NonLinear> Add for Probability<K> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new_unchecked(self.value + rhs.value)
    }
}

// `*`: linear -> probability, others -> raw value
impl Mul for Probability<Linear> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new_unchecked(self.value * rhs.value)
    }
}
impl<K: NonLinear> Mul for Probability<K> {
    type Output = K::Value;
    fn mul(self, rhs: Self) -> K::Value {
        self.value * rhs.value
    }
}

// `-` and `/`: always raw value
impl<K: ProbabilityKind> Sub for Probability<K> {
    type Output = K::Value;
    fn sub(self, rhs: Self) -> K::Value {
        self.value - rhs.value
    }
}
impl<K: ProbabilityKind> Div for Probability<K> {
    type Output = K::Value;
    fn div(self, rhs: Self) -> K::Value {
        self.value / rhs.value
    }
}

// --- comparison against the raw value (C++ got this via implicit conversion) --
// Implemented per concrete kind: a blanket `PartialEq<K::Value>` would overlap
// with the derived `PartialEq<Self>`.

macro_rules! cmp_raw {
    ($($k:ident),*) => {$(
        impl PartialEq<<$k as ProbabilityKind>::Value> for Probability<$k> {
            fn eq(&self, other: &<$k as ProbabilityKind>::Value) -> bool {
                self.value == *other
            }
        }
        impl PartialOrd<<$k as ProbabilityKind>::Value> for Probability<$k> {
            fn partial_cmp(&self, other: &<$k as ProbabilityKind>::Value) -> Option<Ordering> {
                self.value.partial_cmp(other)
            }
        }
    )*};
}
cmp_raw!(Linear, Log, Log10, Phred, HpPhred);

// --- strings -----------------------------------------------------------------

impl<K: ProbabilityKind> fmt::Display for Probability<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

impl<K: ProbabilityKind> FromStr for Probability<K> {
    type Err = ProbabilityError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = s
            .trim()
            .parse::<K::Value>()
            .map_err(|e| ProbabilityError::Parse {
                kind: K::NAME,
                input: s.to_owned(),
                source: Box::new(e),
            })?;
        Self::new(value)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_and_validation() {
        assert!(LinearProbability::new(0.5).is_ok());
        assert!(LinearProbability::new(1.5).is_err());
        assert!(LogProbability::new(0.1).is_err());
        assert!(LogProbability::new(f64::NAN).is_err());
        assert!(HpPhredProbability::new(u16::MAX).is_err());
        assert_eq!(LinearProbability::default().get(), 0.0);
        assert_eq!(PhredProbability::default().get(), 0);
    }

    #[test]
    fn parsing() {
        let p: LinearProbability = "0.25".parse().unwrap();
        assert_eq!(p, 0.25);
        let err = "2".parse::<LinearProbability>().unwrap_err();
        assert!(matches!(err, ProbabilityError::OutOfRange { .. }));
        let err = "abc".parse::<PhredProbability>().unwrap_err();
        assert!(matches!(err, ProbabilityError::Parse { .. }));
        assert!(err.source().is_some());
        assert_eq!(p.to_string(), "0.25");
    }

    #[test]
    fn conversions() {
        let lin = LinearProbability::new(0.001).unwrap();
        let ph: PhredProbability = lin.into();
        assert_eq!(ph.get(), 30);
        let hp: HpPhredProbability = ph.into();
        assert_eq!(hp.get(), 3000);
        let back: LinearProbability = hp.into();
        assert!((back.get() - 0.001).abs() < 1e-12);
        let log = lin.convert::<Log>();
        assert!((log.get() - 0.001f64.ln()).abs() < 1e-12);
        // saturation and out-of-domain inputs
        assert_eq!(
            PhredProbability::from(LinearProbability::new(0.0).unwrap()).get(),
            255
        );
        assert_eq!(
            PhredProbability::from(HpPhredProbability::new(65534).unwrap()).get(),
            255
        );
        assert_eq!(
            PhredProbability::from(LinearProbability::new_unchecked(2.0)).get(),
            0
        );
    }

    #[test]
    fn arithmetic() {
        let a = LinearProbability::new(0.5).unwrap();
        let b = LinearProbability::new(0.2).unwrap();
        let s: f64 = a + b;
        assert!((s - 0.7).abs() < 1e-12);
        let prod: LinearProbability = a * b;
        assert!((prod.get() - 0.1).abs() < 1e-12);
        assert_eq!(a.complement(), 0.5);
        assert!((b.odds_ratio() - 0.25).abs() < 1e-12);

        let x = PhredProbability::new(10).unwrap();
        let y = PhredProbability::new(20).unwrap();
        let sum: PhredProbability = x + y;
        assert_eq!(sum.get(), 30);
        assert!(x.more_probable(y));
        assert!(a.more_probable(b));

        let mut l = LogProbability::new(-1.0).unwrap();
        l += LogProbability::new(-2.0).unwrap();
        assert_eq!(l, -3.0);
        l.scale(-1.0);
        assert_eq!(l, -2.0);
    }

    #[test]
    fn extremes() {
        assert_eq!(PhredProbability::highest().get(), 0);
        assert_eq!(PhredProbability::lowest().get(), 255);
        assert_eq!(LinearProbability::highest().get(), 1.0);
        assert_eq!(LogProbability::lowest().get(), f64::MIN);
    }

    #[test]
    fn tags() {
        let mut p = LinearProbability::default();
        p.set_as_byte_tag(b'A');
        assert!(p.is_tag());
        assert_eq!(p.byte_tag(), Some(b'A'));
        p.set_as_tag();
        assert_eq!(p.byte_tag(), Some(0));

        let mut l = Log10Probability::default();
        l.set_as_byte_tag(7);
        assert!(l.is_tag());
        assert_eq!(l.byte_tag(), Some(7));
        assert_eq!(Log10Probability::new(-1.0).unwrap().byte_tag(), None);

        let mut h = HpPhredProbability::default();
        assert!(!h.is_tag());
        h.set_as_tag();
        assert!(h.is_tag());
    }
}
