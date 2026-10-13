//! Linear indices of multi-dimensional coordinates in row-major and column-major layout.
//!
//! In row-major layout (C, ndarray's default) the last coordinate is contiguous; in column-major
//! layout (Fortran, R) the first.

/// The row-major linear index of `coords` in an array of `shape`.
///
/// ```
/// use coretools_rs::stats::linear_index;
///
/// assert_eq!(linear_index(&[1, 2, 3], &[2, 3, 4]), 23);
/// ```
///
/// # Panics
///
/// If there is not one coordinate per dimension. Coordinates beyond the shape are a dev error,
/// checked in debug builds.
pub fn linear_index(coords: &[usize], shape: &[usize]) -> usize {
    check_coords(coords, shape);
    coords
        .iter()
        .zip(shape)
        .fold(0, |index, (&coord, &len)| index * len + coord)
}

/// The column-major linear index of `coords` in an array of `shape`.
///
/// ```
/// use coretools_rs::stats::linear_index_col_major;
///
/// assert_eq!(linear_index_col_major(&[1, 2, 0], &[2, 3, 4]), 5);
/// ```
///
/// # Panics
///
/// As [`linear_index`].
pub fn linear_index_col_major(coords: &[usize], shape: &[usize]) -> usize {
    check_coords(coords, shape);
    coords
        .iter()
        .zip(shape)
        .rev()
        .fold(0, |index, (&coord, &len)| index * len + coord)
}

/// The coordinates of the row-major linear `index` in an array of `shape`: the inverse of
/// [`linear_index`].
///
/// # Panics
///
/// An index beyond the shape is a dev error, checked in debug builds.
pub fn subscripts(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    fill_row_major(index, shape, &mut coords);
    coords
}

/// [`subscripts`] for a shape of fixed dimension.
///
/// ```
/// use coretools_rs::stats::subscripts_array;
///
/// assert_eq!(subscripts_array(23, &[2, 3, 4]), [1, 2, 3]);
/// ```
pub fn subscripts_array<const N: usize>(index: usize, shape: &[usize; N]) -> [usize; N] {
    let mut coords = [0; N];
    fill_row_major(index, shape, &mut coords);
    coords
}

/// The coordinates of the column-major linear `index` in an array of `shape`: the inverse of
/// [`linear_index_col_major`].
///
/// # Panics
///
/// As [`subscripts`].
pub fn subscripts_col_major(index: usize, shape: &[usize]) -> Vec<usize> {
    check_index(index, shape);
    let mut coords = vec![0; shape.len()];
    fill(index, shape.iter().zip(coords.iter_mut()));
    coords
}

fn fill_row_major(index: usize, shape: &[usize], coords: &mut [usize]) {
    check_index(index, shape);
    fill(index, shape.iter().zip(coords.iter_mut()).rev());
}

/// Peels the coordinates off `index`, innermost dimension first.
fn fill<'a>(mut index: usize, dimensions: impl Iterator<Item = (&'a usize, &'a mut usize)>) {
    for (&len, coord) in dimensions {
        *coord = index % len;
        index /= len;
    }
}

#[track_caller]
fn check_coords(coords: &[usize], shape: &[usize]) {
    assert_eq!(
        coords.len(),
        shape.len(),
        "{} coordinates for a shape of {} dimensions",
        coords.len(),
        shape.len()
    );
    debug_assert!(
        coords.iter().zip(shape).all(|(coord, len)| coord < len),
        "{coords:?} lies outside the shape {shape:?}"
    );
}

#[track_caller]
fn check_index(index: usize, shape: &[usize]) {
    debug_assert!(
        shape
            .iter()
            .try_fold(1_usize, |len, &dim| len.checked_mul(dim))
            .is_none_or(|len| index < len),
        "index {index} lies outside the shape {shape:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every coordinate of `shape`, in row-major order.
    fn all_coords(shape: &[usize]) -> Vec<Vec<usize>> {
        shape.iter().fold(vec![vec![]], |prefixes, &len| {
            prefixes
                .iter()
                .flat_map(|prefix| {
                    (0..len).map(move |i| {
                        let mut coords = prefix.clone();
                        coords.push(i);
                        coords
                    })
                })
                .collect()
        })
    }

    const SHAPES: [&[usize]; 7] = [
        &[1],
        &[5],
        &[1, 5],
        &[5, 1],
        &[5, 3],
        &[2, 3, 4],
        &[2, 3, 4, 5],
    ];

    #[test]
    fn row_major_index_has_the_last_coordinate_contiguous() {
        let expected = |c: &[usize], s: &[usize]| match *c {
            [i] => i,
            [i, j] => i * s[1] + j,
            [i, j, k] => k + 4 * (j + 3 * i),
            [i, j, k, l] => l + 5 * (k + 4 * (j + 3 * i)),
            _ => unreachable!(),
        };
        for shape in SHAPES {
            for coords in all_coords(shape) {
                assert_eq!(
                    linear_index(&coords, shape),
                    expected(&coords, shape),
                    "{coords:?} in {shape:?}"
                );
            }
        }
    }

    #[test]
    fn column_major_index_has_the_first_coordinate_contiguous() {
        let expected = |c: &[usize], s: &[usize]| match *c {
            [i] => i,
            [i, j] => i + j * s[0],
            [i, j, k] => i + 2 * (j + 3 * k),
            [i, j, k, l] => i + 2 * (j + 3 * (k + 4 * l)),
            _ => unreachable!(),
        };
        for shape in SHAPES {
            for coords in all_coords(shape) {
                assert_eq!(
                    linear_index_col_major(&coords, shape),
                    expected(&coords, shape),
                    "{coords:?} in {shape:?}"
                );
            }
        }
    }

    #[test]
    fn subscripts_invert_the_linear_index() {
        for shape in SHAPES {
            let len = shape.iter().product();
            for index in 0..len {
                assert_eq!(linear_index(&subscripts(index, shape), shape), index);
                assert_eq!(
                    linear_index_col_major(&subscripts_col_major(index, shape), shape),
                    index
                );
            }
        }
        for index in 0..24 {
            let coords = subscripts_array(index, &[2, 3, 4]);
            assert_eq!(coords.to_vec(), subscripts(index, &[2, 3, 4]));
        }
        assert_eq!(subscripts(23, &[2, 3, 4]), [1, 2, 3]);
        assert_eq!(subscripts_col_major(23, &[2, 3, 4]), [1, 2, 3]);
        assert_eq!(subscripts(5, &[2, 3, 4]), [0, 1, 1]);
        assert_eq!(subscripts_col_major(5, &[2, 3, 4]), [1, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "3 coordinates for a shape of 2 dimensions")]
    fn coordinates_need_one_per_dimension() {
        linear_index(&[0, 0, 0], &[5, 3]);
    }

    #[test]
    #[should_panic(expected = "1 coordinates for a shape of 2 dimensions")]
    fn column_major_coordinates_need_one_per_dimension() {
        linear_index_col_major(&[0], &[5, 3]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "[0, 3] lies outside the shape [5, 3]")]
    fn coordinates_beyond_the_shape_are_a_dev_error() {
        linear_index(&[0, 3], &[5, 3]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "index 24 lies outside the shape [2, 3, 4]")]
    fn an_index_beyond_the_shape_is_a_dev_error() {
        subscripts(24, &[2, 3, 4]);
    }
}
