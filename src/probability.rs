use core::cmp::Ordering;
use core::error::Error;
use core::f64::consts::{LN_10, LOG10_E};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};
use core::str::FromStr;

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

// --- phred-only --------------------------------------------------------------

impl Probability<Phred> {
    /// Lowest ASCII base-quality character (`!`, Phred 0).
    const ASCII_MIN: u8 = 33;
    /// Highest ASCII base-quality character (`~`, Phred 93).
    const ASCII_MAX: u8 = 126;

    /// The ASCII base-quality character (Phred + 33), saturating at `~` (Phred 93).
    pub fn to_ascii(self) -> u8 {
        self.value.min(Self::ASCII_MAX - Self::ASCII_MIN) + Self::ASCII_MIN
    }

    /// Parse an ASCII base-quality character (Phred + 33); bytes outside
    /// `!..=~` are clamped into that range.
    pub fn from_ascii(byte: u8) -> Self {
        Self::new_unchecked(byte.clamp(Self::ASCII_MIN, Self::ASCII_MAX) - Self::ASCII_MIN)
    }
}

// --- sets of probabilities ---------------------------------------------------

impl<K: ProbabilityKind> Probability<K> {
    /// Sum of the probabilities, as a plain linear value (may exceed 1).
    pub fn sum<I: IntoIterator<Item = Self>>(ps: I) -> f64
    where
        Linear: ConvertFrom<K>,
    {
        ps.into_iter().map(|p| p.convert::<Linear>().get()).sum()
    }

    /// Arithmetic mean of the probabilities, in this representation; `None` if `ps` is empty.
    ///
    /// Non-linear representations are averaged in log space (log-sum-exp), so
    /// log probabilities far below `f64` underflow keep their precision.
    pub fn average<I: IntoIterator<Item = Self>>(ps: I) -> Option<Self>
    where
        Linear: ConvertFrom<K>,
        Log: ConvertFrom<K>,
        K: ConvertFrom<Linear> + ConvertFrom<Log>,
    {
        let mut n = 0_usize;
        if K::IS_LINEAR {
            let total: f64 = ps
                .into_iter()
                .inspect(|_| n += 1)
                .map(|p| p.convert::<Linear>().get())
                .sum();
            return (n > 0)
                .then(|| Probability::<Linear>::new_unchecked(total / n as f64).convert());
        }
        // Streaming log-sum-exp: total = exp(max) * scaled.
        let mut max = f64::NEG_INFINITY;
        let mut scaled = 0.0;
        for x in ps.into_iter().map(|p| p.convert::<Log>().get()) {
            n += 1;
            if x > max {
                scaled = scaled * (max - x).exp() + 1.0;
                max = x;
            } else if x > f64::NEG_INFINITY {
                scaled += (x - max).exp();
            }
        }
        (n > 0).then(|| {
            Probability::<Log>::new_unchecked(max + scaled.ln() - (n as f64).ln()).convert()
        })
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

    /// Relative tolerance of the port (ADR-0001), absolute near zero.
    #[track_caller]
    fn assert_close(actual: f64, expected: f64) {
        let tol = 1e-9 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tol,
            "{actual} is not within {tol} of {expected}"
        );
    }

    fn lin(p: f64) -> LinearProbability {
        LinearProbability::new(p).unwrap()
    }
    fn log(p: f64) -> LogProbability {
        LogProbability::new(p).unwrap()
    }
    fn log10(p: f64) -> Log10Probability {
        Log10Probability::new(p).unwrap()
    }
    fn phred(p: u8) -> PhredProbability {
        PhredProbability::new(p).unwrap()
    }
    fn hp_phred(p: u16) -> HpPhredProbability {
        HpPhredProbability::new(p).unwrap()
    }

    // --- ported from C++ TProbabilityTests.cpp -------------------------------
    // Not ported: arithmetic with a bare `double` (C++ implicit decay; Rust
    // callers use `get()`), the `P`/`logP`/`log10P` helpers (`new`/`From`), and
    // `static_assert`s that mixed representations or `+=` on Linear and
    // `setAsTag` on Phred do not compile (enforced by the type system).

    #[test]
    fn linear_probability() {
        assert!(LinearProbability::new(1.2).is_err());
        assert!(LinearProbability::new(-0.1).is_err());
        assert!(LinearProbability::new(0.1).is_ok());

        let p01 = lin(0.1);
        let p05 = lin(0.5);
        assert_eq!(p01.get(), 0.1);
        assert_eq!(p05, 0.5);
        assert_close(p01.complement().get(), 0.9);
        assert_close(p05.complement().get(), 0.5);
        assert_close(p01.odds_ratio(), 0.1 / 0.9);
        assert_close(p05.odds_ratio(), 1.0);

        assert_eq!("1.".parse::<LinearProbability>().unwrap(), 1.0);
        assert_eq!("0.5".parse::<LinearProbability>().unwrap(), 0.5);

        assert_ne!(p01, p05);
        assert!(p01 < p05);
        assert!(p01 < 1.5);
        assert!(p01 != 1.5);

        let mut p = p05;
        p *= p01;
        assert_close(p.get(), 0.05);
        assert_eq!(p, p05 * p01);
        let p = p01 * p01;
        assert_close(p / p01, 0.1);
        assert!(LinearProbability::new(p05 / p).is_err());

        assert_close(LinearProbability::new(p01 / p05).unwrap().get(), 0.2);
        let p06 = LinearProbability::new(p01 + p05).unwrap();
        assert_close(p06.get(), 0.6);
        assert!(LinearProbability::new(p06 + p06).is_err());
        assert_close(LinearProbability::new(p05 - p01).unwrap().get(), 0.4);
        assert!(LinearProbability::new(p01 - p05).is_err());

        assert_eq!(LinearProbability::lowest(), 0.0);
        assert_eq!(LinearProbability::highest(), 1.0);

        let mut p = lin(0.6);
        p.scale(0.9);
        assert_close(p.get(), 0.6 / 0.9);
        assert_close(lin(0.5).scale(2.0).get(), 0.25);
    }

    /// Shared body of the C++ `TLogProbability` and `TLog10Probability` tests.
    fn log_like_probability<K: NonLinear<Value = f64>>() {
        let new = |v| Probability::<K>::new(v);
        assert!(new(1.2).is_err());
        assert!(new(-0.1).is_ok());
        assert!(new(0.0).is_ok());

        let lp5 = new(-5.0).unwrap();
        let lp4 = new(-4.0).unwrap();
        assert_eq!(lp5.get(), -5.0);
        assert_ne!(lp5, lp4);
        assert!(lp5 < lp4);
        assert!(lp5.get() < 1.5);
        assert_eq!(
            "0".parse::<Probability<K>>().unwrap(),
            Probability::new_unchecked(0.0)
        );

        let mut lp = lp5;
        lp += lp5;
        assert_eq!(lp.get(), -10.0);
        assert_eq!((lp5 + lp4).get(), -9.0);

        assert!(new(new(-5.0).unwrap() - new(-6.0).unwrap()).is_err());
        assert_eq!(new(lp5 - lp4).unwrap().get(), -1.0);
        assert!(new(lp4 - lp5).is_err());

        assert_eq!(Probability::<K>::lowest().get(), f64::MIN);
        assert_eq!(Probability::<K>::highest().get(), 0.0);

        let mut lp = lp5;
        lp.scale(-4.0);
        assert_eq!(lp.get(), -1.0);
        assert_eq!(new(-10.0).unwrap().scale(5.0).get(), -15.0);
    }

    #[test]
    fn log_probability() {
        log_like_probability::<Log>();
    }

    #[test]
    fn log10_probability() {
        log_like_probability::<Log10>();
    }

    /// C++ `CHECK_INTERVALS` threw; here an out-of-range `scale` is a dev error.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "linear probability cannot have value")]
    fn linear_scale_out_of_range_is_a_dev_error() {
        lin(0.5).scale(0.1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "log probability cannot have value")]
    fn log_scale_out_of_range_is_a_dev_error() {
        log(-10.0).scale(-11.0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "log10 probability cannot have value")]
    fn log10_scale_out_of_range_is_a_dev_error() {
        log10(-10.0).scale(-11.0);
    }

    #[test]
    fn conversions() {
        // Probability 0: the C++ compared `ln(0)` with `lowest()` only after a
        // lossy cast to float; the actual value is -inf in both languages.
        let zero = LinearProbability::default();
        assert_eq!(LogProbability::from(zero).get(), f64::NEG_INFINITY);
        assert_eq!(Log10Probability::from(zero).get(), f64::NEG_INFINITY);
        assert_eq!(PhredProbability::from(zero), PhredProbability::lowest());
        assert_eq!(HpPhredProbability::from(zero), HpPhredProbability::lowest());

        for (p, ph, hp) in [(0.2, 7, 699), (0.3, 5, 523)] {
            let p = lin(p);
            assert_close(LogProbability::from(p).get(), p.get().ln());
            assert_close(Log10Probability::from(p).get(), p.get().log10());
            assert_eq!(PhredProbability::from(p), ph);
            assert_eq!(HpPhredProbability::from(p), hp);
        }

        for (lp, ph, hp) in [(-1.0, 4, 434), (-2.0, 9, 869)] {
            let lp = log(lp);
            assert_close(LinearProbability::from(lp).get(), lp.get().exp());
            assert_close(Log10Probability::from(lp).get(), lp.get().exp().log10());
            assert_eq!(PhredProbability::from(lp), ph);
            assert_eq!(HpPhredProbability::from(lp), hp);
        }

        for (l10p, ph, hp) in [(-3.0, 30, 3000), (-4.123, 41, 4123)] {
            let l10p = log10(l10p);
            assert_close(LinearProbability::from(l10p).get(), 10f64.powf(l10p.get()));
            assert_close(
                LogProbability::from(l10p).get(),
                10f64.powf(l10p.get()).ln(),
            );
            assert_eq!(PhredProbability::from(l10p), ph);
            assert_eq!(HpPhredProbability::from(l10p), hp);
        }

        let ph = phred(11);
        assert_close(LinearProbability::from(ph).get(), 10f64.powf(-1.1));
        assert_close(LogProbability::from(ph).get(), 10f64.powf(-1.1).ln());
        assert_close(Log10Probability::from(ph).get(), -1.1);
        assert_eq!(HpPhredProbability::from(ph), 1100);

        let hp = hp_phred(12345);
        assert_close(LinearProbability::from(hp).get(), 10f64.powf(-12.345));
        assert_close(LogProbability::from(hp).get(), 10f64.powf(-12.345).ln());
        assert_close(Log10Probability::from(hp).get(), -12.345);
        assert_eq!(PhredProbability::from(hp), 123);
    }

    #[test]
    fn arithmetic_result_representations() {
        // `*`: Linear stays a probability, the others yield the raw value.
        let p: LinearProbability = lin(0.5) * lin(0.4);
        assert_close(p.get(), 0.2);
        let raw: f64 = log(-2.0) * log(-3.0);
        assert_eq!(raw, 6.0);
        let raw: f64 = log10(-2.0) * log10(-3.0);
        assert_eq!(raw, 6.0);
        let raw: u8 = phred(2) * phred(3);
        assert_eq!(raw, 6);
        let raw: u16 = hp_phred(2) * hp_phred(3);
        assert_eq!(raw, 6);

        // `+`: Linear yields the raw value, the others stay probabilities.
        let raw: f64 = lin(0.5) + lin(0.25);
        assert_eq!(raw, 0.75);
        let lp: LogProbability = log(-2.0) + log(-3.0);
        assert_eq!(lp, -5.0);
        let l10p: Log10Probability = log10(-2.0) + log10(-3.0);
        assert_eq!(l10p, -5.0);
        let ph: PhredProbability = phred(10) + phred(20);
        assert_eq!(ph, 30);
        let hp: HpPhredProbability = hp_phred(1000) + hp_phred(2000);
        assert_eq!(hp, 3000);

        // `+=` on every non-linear representation.
        let mut ph = phred(10);
        ph += phred(20);
        assert_eq!(ph, 30);
        let mut hp = hp_phred(1000);
        hp += hp_phred(2000);
        assert_eq!(hp, 3000);

        // `-` and `/`: always the raw value.
        let raw: f64 = lin(0.5) - lin(0.25);
        assert_eq!(raw, 0.25);
        let raw: f64 = log(-2.0) - log(-3.0);
        assert_eq!(raw, 1.0);
        let raw: u8 = phred(30) - phred(10);
        assert_eq!(raw, 20);
        let raw: u16 = hp_phred(3000) / hp_phred(1000);
        assert_eq!(raw, 3);
        let raw: f64 = log10(-6.0) / log10(-3.0);
        assert_eq!(raw, 2.0);
    }

    #[test]
    fn base_quality_round_trips_through_every_representation() {
        assert_eq!(PhredProbability::highest().to_ascii(), 33);
        assert_eq!(PhredProbability::lowest().to_ascii(), 126);
        for q in 33..=126 {
            let ph = PhredProbability::from_ascii(q);
            assert_eq!(ph.to_ascii(), q);
            assert_eq!(
                PhredProbability::from(HpPhredProbability::from(ph)).to_ascii(),
                q
            );
            assert_eq!(
                PhredProbability::from(LinearProbability::from(ph)).to_ascii(),
                q
            );
            assert_eq!(
                PhredProbability::from(LogProbability::from(ph)).to_ascii(),
                q
            );
            assert_eq!(
                PhredProbability::from(Log10Probability::from(ph)).to_ascii(),
                q
            );
        }
    }

    #[test]
    fn phred_round_trips_through_every_representation() {
        for i in PhredProbability::min()..PhredProbability::max() {
            let ph = phred(i);
            let hp = HpPhredProbability::from(ph);
            assert_eq!(hp, u16::from(i) * 100);
            assert_eq!(PhredProbability::from(hp), ph);

            let p = LinearProbability::from(ph);
            assert_eq!(PhredProbability::from(p), ph);
            assert_eq!(HpPhredProbability::from(p), hp);
            let lp = LogProbability::from(ph);
            assert_eq!(PhredProbability::from(lp), ph);
            assert_eq!(HpPhredProbability::from(lp), hp);
            let l10p = Log10Probability::from(ph);
            assert_eq!(PhredProbability::from(l10p), ph);
            assert_eq!(HpPhredProbability::from(l10p), hp);
        }
    }

    #[test]
    fn average_of_linear_probabilities() {
        let avg = |ps: &[f64]| {
            LinearProbability::average(ps.iter().map(|&p| lin(p)))
                .unwrap()
                .get()
        };
        assert_close(avg(&[0.5]), 0.5);
        assert_close(avg(&[0.5, 0.5]), 0.5);
        assert_close(avg(&[0.5, 0.5, 0.5]), 0.5);
        assert_close(avg(&[0.5, 0.5, 0.5, 0.5]), 0.5);
        assert_close(avg(&[0.5, 1.0, 0.0]), 0.5);
        assert_close(avg(&[0.5, 0.0]), 0.25);
    }

    #[test]
    fn tags() {
        let mut p = LinearProbability::default();
        let mut lp = LogProbability::default();
        let mut l10p = Log10Probability::default();
        let mut hp = HpPhredProbability::default();
        assert!(!p.is_tag() && !lp.is_tag() && !l10p.is_tag() && !hp.is_tag());

        p.set_as_tag();
        lp.set_as_tag();
        l10p.set_as_tag();
        hp.set_as_tag();
        assert!(p.is_tag() && lp.is_tag() && l10p.is_tag() && hp.is_tag());
    }

    // --- Rust-side behaviour -------------------------------------------------

    #[test]
    fn construction_rejects_nan_and_the_reserved_tag() {
        assert!(LogProbability::new(f64::NAN).is_err());
        assert!(LinearProbability::new(f64::NAN).is_err());
        assert!(HpPhredProbability::new(u16::MAX).is_err());
        assert_eq!(LinearProbability::default(), 0.0);
        assert_eq!(PhredProbability::default(), 0);
    }

    #[test]
    fn parsing_errors_distinguish_syntax_from_range() {
        let err = "2".parse::<LinearProbability>().unwrap_err();
        assert!(matches!(err, ProbabilityError::OutOfRange { .. }));
        let err = "abc".parse::<PhredProbability>().unwrap_err();
        assert!(matches!(err, ProbabilityError::Parse { .. }));
        assert!(err.source().is_some());
        assert_eq!(lin(0.25).to_string(), "0.25");
    }

    #[test]
    fn conversions_to_phred_saturate() {
        assert_eq!(PhredProbability::from(hp_phred(65534)), 255);
        assert_eq!(PhredProbability::from(log(-1000.0)), 255);
        assert_eq!(HpPhredProbability::from(log10(-1000.0)), 65534);
        // Out-of-domain input (C++: undefined behaviour) maps to the highest probability.
        assert_eq!(
            PhredProbability::from(LinearProbability::new_unchecked(2.0)),
            0
        );
    }

    #[test]
    fn more_probable_respects_inverted_representations() {
        assert!(phred(10).more_probable(phred(20)));
        assert!(hp_phred(10).more_probable(hp_phred(20)));
        assert!(lin(0.5).more_probable(lin(0.2)));
        assert!(log(-1.0).more_probable(log(-2.0)));
    }

    #[test]
    fn byte_tags() {
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
        assert_eq!(log10(-1.0).byte_tag(), None);
    }

    /// Divergence: C++ floored (integer division by 100).
    #[test]
    fn hp_phred_to_phred_rounds_to_nearest() {
        assert_eq!(PhredProbability::from(hp_phred(699)), 7);
        assert_eq!(PhredProbability::from(hp_phred(150)), 2);
        assert_eq!(PhredProbability::from(hp_phred(149)), 1);
    }

    #[test]
    fn phred_ascii_base_quality_clamps() {
        assert_eq!(phred(30).to_ascii(), b'?');
        assert_eq!(phred(94).to_ascii(), b'~');
        assert_eq!(PhredProbability::from_ascii(b'I'), 40);
        assert_eq!(PhredProbability::from_ascii(b' '), 0);
        assert_eq!(PhredProbability::from_ascii(0), 0);
        assert_eq!(PhredProbability::from_ascii(127), 93);
        assert_eq!(PhredProbability::from_ascii(u8::MAX), 93);
    }

    #[test]
    fn sum_adds_probabilities_in_linear_space() {
        assert_close(
            LinearProbability::sum([lin(0.5), lin(0.25), lin(0.75)]),
            1.5,
        );
        assert_close(
            LogProbability::sum([log(0.5f64.ln()), log(0.25f64.ln())]),
            0.75,
        );
        assert_close(PhredProbability::sum([phred(10), phred(20)]), 0.11);
        assert_eq!(Log10Probability::sum([]), 0.0);
    }

    #[test]
    fn average_stays_in_the_same_representation() {
        assert_close(
            Log10Probability::average([log10(-1.0), log10(-2.0)])
                .unwrap()
                .get(),
            0.055f64.log10(),
        );
        assert_eq!(
            PhredProbability::average([phred(10), phred(20)]).unwrap(),
            13
        ); // 12.596
        assert_eq!(
            HpPhredProbability::average([hp_phred(0), hp_phred(3000)]).unwrap(),
            301
        ); // 300.6
        assert!(LogProbability::average([]).is_none());
        assert!(LinearProbability::average([]).is_none());
    }

    #[test]
    fn average_of_log_probabilities_does_not_underflow() {
        let expected = -1000.0 + ((1.0 + (-1.0f64).exp()) / 2.0).ln();
        assert_close(
            LogProbability::average([log(-1000.0), log(-1001.0)])
                .unwrap()
                .get(),
            expected,
        );
    }
}
