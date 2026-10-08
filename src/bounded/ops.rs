//! Arithmetic between two values of the same `Bounded` type (see the table in the module docs).

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Bounded;
use super::interval::{
    Negative, Positive, StrictlyNegative, StrictlyPositive, Unbounded, ZeroOneClosed, ZeroOneOpen,
    ZeroOpenOneClosed,
};

/// The result stays in the interval: `Output = Self`, plus the compound assignment.
macro_rules! bounded_op {
    ($t:ty: $($i:ident),+ => $Op:ident::$op:ident, $OpAssign:ident::$op_assign:ident,
     |$a:ident, $b:ident| $body:expr) => {$(
        impl $Op for Bounded<$t, $i> {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn $op(self, rhs: Self) -> Self {
                let ($a, $b) = (self.value, rhs.value);
                Self::new_unchecked($body)
            }
        }
        impl $OpAssign for Bounded<$t, $i> {
            #[inline]
            #[track_caller]
            fn $op_assign(&mut self, rhs: Self) {
                *self = $Op::$op(*self, rhs);
            }
        }
    )+};
}

/// The result may leave the interval: `Output` is the raw number.
macro_rules! raw_op {
    ($t:ty: $($i:ident),+ => $Op:ident::$op:ident, |$a:ident, $b:ident| $body:expr) => {$(
        impl $Op for Bounded<$t, $i> {
            type Output = $t;
            #[inline]
            #[track_caller]
            fn $op(self, rhs: Self) -> $t {
                let ($a, $b) = (self.value, rhs.value);
                $body
            }
        }
    )+};
}

// --- f64 ----------------------------------------------------------------------
// Float overflow, underflow and division by zero can still carry a result out of
// the interval (e.g. a product of two tiny `ZeroOneOpen` values rounding to 0);
// `new_unchecked` catches that in debug builds.

bounded_op!(f64: Unbounded, Positive, StrictlyPositive, Negative, StrictlyNegative
    => Add::add, AddAssign::add_assign, |a, b| a + b);
raw_op!(f64: ZeroOneClosed, ZeroOneOpen, ZeroOpenOneClosed => Add::add, |a, b| a + b);

bounded_op!(f64: Unbounded => Sub::sub, SubAssign::sub_assign, |a, b| a - b);
raw_op!(f64: Positive, StrictlyPositive, Negative, StrictlyNegative, ZeroOneClosed, ZeroOneOpen,
    ZeroOpenOneClosed => Sub::sub, |a, b| a - b);

bounded_op!(f64: Unbounded, Positive, StrictlyPositive, ZeroOneClosed, ZeroOneOpen, ZeroOpenOneClosed
    => Mul::mul, MulAssign::mul_assign, |a, b| a * b);
raw_op!(f64: Negative, StrictlyNegative => Mul::mul, |a, b| a * b);

bounded_op!(f64: Unbounded, Positive, StrictlyPositive => Div::div, DivAssign::div_assign, |a, b| a / b);
raw_op!(f64: Negative, StrictlyNegative, ZeroOneClosed, ZeroOneOpen, ZeroOpenOneClosed
    => Div::div, |a, b| a / b);

impl Neg for Bounded<f64, Unbounded> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new_unchecked(-self.value)
    }
}

macro_rules! raw_neg {
    ($($i:ident),+) => {$(
        impl Neg for Bounded<f64, $i> {
            type Output = f64;
            #[inline]
            fn neg(self) -> f64 {
                -self.value
            }
        }
    )+};
}
raw_neg!(
    Positive,
    StrictlyPositive,
    Negative,
    StrictlyNegative,
    ZeroOneClosed,
    ZeroOneOpen,
    ZeroOpenOneClosed
);

// --- unsigned integers ----------------------------------------------------------
// Overflow and subtraction below 0 panic in every build: unlike float rounding,
// a wrapped integer is silently far from the true result. Every result that is
// returned therefore lies in `[0, max]`.

#[cold]
#[track_caller]
fn out_of_range(a: impl fmt::Display, op: char, b: impl fmt::Display, t: &str) -> ! {
    panic!("{a} {op} {b} is outside the range of {t}")
}

// Bodies use `match`, not `unwrap_or_else`: a closure would hide the caller's
// location from `#[track_caller]`.
macro_rules! unsigned {
    ($($t:ident),+) => {$(
        bounded_op!($t: Positive, StrictlyPositive => Add::add, AddAssign::add_assign,
            |a, b| match a.checked_add(b) {
                Some(v) => v,
                None => out_of_range(a, '+', b, stringify!($t)),
            });
        bounded_op!($t: Positive, StrictlyPositive => Mul::mul, MulAssign::mul_assign,
            |a, b| match a.checked_mul(b) {
                Some(v) => v,
                None => out_of_range(a, '*', b, stringify!($t)),
            });
        // `a - a` is 0: inside [0, max], outside (0, max].
        bounded_op!($t: Positive => Sub::sub, SubAssign::sub_assign,
            |a, b| match a.checked_sub(b) {
                Some(v) => v,
                None => out_of_range(a, '-', b, stringify!($t)),
            });
        raw_op!($t: StrictlyPositive => Sub::sub,
            |a, b| match a.checked_sub(b) {
                Some(v) => v,
                None => out_of_range(a, '-', b, stringify!($t)),
            });
        // Integer division rounds towards 0: inside [0, max], outside (0, max].
        bounded_op!($t: Positive => Div::div, DivAssign::div_assign, |a, b| a / b);
        raw_op!($t: StrictlyPositive => Div::div, |a, b| a / b);
    )+};
}
unsigned!(u8, u16, u32, u64, usize);
