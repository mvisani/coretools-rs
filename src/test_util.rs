/// Relative tolerance of the port (issue #1: ~1e-9); exact when `expected` is 0.
#[track_caller]
pub(crate) fn assert_close(actual: f64, expected: f64) {
    let tol = 1e-9 * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{actual} is not within {tol} of {expected}"
    );
}
