/// Relative tolerance of the port (issue #1: ~1e-9); exact when `expected` is 0.
#[track_caller]
pub(crate) fn assert_close(actual: f64, expected: f64) {
    let tol = 1e-9 * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{actual} is not within {tol} of {expected}"
    );
}

/// gtest's `EXPECT_FLOAT_EQ`: both values rounded to `f32` are at most 4 ULPs apart. For C++ tests
/// whose expected values have only single-precision digits.
#[track_caller]
pub(crate) fn assert_float_eq(actual: f64, expected: f64) {
    // Map the f32 bit patterns onto a monotone integer line so that ULP distance is a difference.
    fn ordered(x: f32) -> i64 {
        let bits = i64::from(x.to_bits());
        if bits & 0x8000_0000 == 0 {
            bits
        } else {
            0x8000_0000 - bits
        }
    }
    // rounding to f32 is the comparison
    let ulps = (ordered(actual as f32) - ordered(expected as f32)).abs();
    assert!(
        ulps <= 4,
        "{actual} is {ulps} f32 ULPs from {expected} (at most 4 allowed)"
    );
}
