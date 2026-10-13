use core::ops::AddAssign;

use ndarray::{Array, Dimension, NdIndex, ShapeBuilder, Zip};

use super::{
    CountDistribution, combined_of, max_of, min_of, ratio, sum_of, total_above_of, total_of,
};

/// A [`CountDistribution`] per cell of an N-dimensional array of fixed shape.
///
/// Summaries such as [`total`](Self::total), [`mean`](Self::mean) or [`max`](Self::max) are over the
/// values of all cells.
///
/// ```
/// use coretools_rs::stats::CountDistributionArray;
///
/// let mut depth_per_sample_and_chunk = CountDistributionArray::new([2, 3]);
/// depth_per_sample_and_chunk.add([1, 2], 7);
/// assert_eq!(depth_per_sample_and_chunk.get([1, 2]).unwrap().count(7), 1);
/// assert_eq!(depth_per_sample_and_chunk.max(), Some(7));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountDistributionArray<D: Dimension> {
    distributions: Array<CountDistribution, D>,
}

impl<D: Dimension> CountDistributionArray<D> {
    /// Cells of the given shape, without counts.
    pub fn new<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self {
        Self {
            distributions: Array::default(shape),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.distributions.shape()
    }

    /// The number of cells.
    pub fn len(&self) -> usize {
        self.distributions.len()
    }

    /// Whether the array has no cells.
    pub fn is_empty(&self) -> bool {
        self.distributions.is_empty()
    }

    /// The distribution of the cell at `index`; `None` outside the shape.
    pub fn get<I: NdIndex<D>>(&self, index: I) -> Option<&CountDistribution> {
        self.distributions.get(index)
    }

    /// The cells as an ndarray array.
    pub fn as_array(&self) -> &Array<CountDistribution, D> {
        &self.distributions
    }

    /// Counts one occurrence of `value` in the cell at `index`.
    ///
    /// # Panics
    ///
    /// If `index` lies outside the shape.
    pub fn add<I: NdIndex<D>>(&mut self, index: I, value: usize) {
        self.distributions[index].add(value);
    }

    /// Counts `n` occurrences of `value` in the cell at `index`.
    ///
    /// # Panics
    ///
    /// If `index` lies outside the shape.
    pub fn add_n<I: NdIndex<D>>(&mut self, index: I, value: usize, n: u64) {
        self.distributions[index].add_n(value, n);
    }

    /// Adds the counts of `dist` to the cell at `index`.
    ///
    /// # Panics
    ///
    /// If `index` lies outside the shape.
    pub fn add_distribution<I: NdIndex<D>>(&mut self, index: I, dist: &CountDistribution) {
        self.distributions[index] += dist;
    }

    /// The number of occurrences of all values of all cells.
    pub fn total(&self) -> u64 {
        total_of(self.distributions.iter())
    }

    /// The number of occurrences of values above `value` over all cells.
    pub fn total_above(&self, value: usize) -> u64 {
        total_above_of(self.distributions.iter(), value)
    }

    /// The sum of all occurrences of all cells.
    pub fn sum(&self) -> u64 {
        sum_of(self.distributions.iter())
    }

    /// The mean of all occurrences of all cells; 0 without counts.
    pub fn mean(&self) -> f64 {
        ratio(self.sum(), self.total())
    }

    /// The fraction of occurrences above `value` over all cells; 0 without counts.
    pub fn fraction_above(&self, value: usize) -> f64 {
        ratio(self.total_above(value), self.total())
    }

    /// The smallest value that occurred in any cell; `None` without counts.
    pub fn min(&self) -> Option<usize> {
        min_of(self.distributions.iter())
    }

    /// The largest value that occurred in any cell; `None` without counts.
    pub fn max(&self) -> Option<usize> {
        max_of(self.distributions.iter())
    }

    /// The distribution of the values of all cells together.
    pub fn combined(&self) -> CountDistribution {
        combined_of(self.distributions.iter())
    }
}

impl<D: Dimension> AddAssign<&CountDistributionArray<D>> for CountDistributionArray<D> {
    /// Adds the distribution of every cell of `other` to the same cell.
    ///
    /// # Panics
    ///
    /// If the shapes differ.
    fn add_assign(&mut self, other: &CountDistributionArray<D>) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "cannot add a count distribution array of shape {:?} to one of shape {:?}",
            other.shape(),
            self.shape()
        );
        Zip::from(&mut self.distributions)
            .and(&other.distributions)
            .for_each(|dist, other| *dist += other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_to_the_distribution_of_each_cell() {
        let mut array = CountDistributionArray::new([2, 3, 4]);
        let mut sum = 0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    array.add([i, j, k], i * j * k);
                    sum += (i * j * k) as u64;
                    assert_eq!(array.get([i, j, k]).unwrap().max(), Some(i * j * k));
                }
            }
        }
        assert_eq!(array.len(), 24);
        assert_eq!(array.shape(), [2, 3, 4]);
        assert_eq!(array.total(), 24);
        assert_eq!(array.sum(), sum);
        assert_eq!(array.mean(), sum as f64 / 24.0);
        assert_eq!(array.max(), Some(6));
        assert!(array.get([2, 0, 0]).is_none());
    }

    #[test]
    fn summaries_over_all_cells() {
        let mut array = CountDistributionArray::new([2, 2]);
        array.add_n([0, 1], 3, 4);
        array.add([1, 1], 7);
        let mut dist = CountDistribution::new();
        dist.add_n(1, 5);
        array.add_distribution([1, 0], &dist);

        assert_eq!(array.total(), 10);
        assert_eq!(array.total_above(3), 1);
        assert_eq!(array.fraction_above(1), 0.5);
        assert_eq!(array.sum(), 24);
        assert_eq!(array.min(), Some(1));
        assert_eq!(array.max(), Some(7));
        let combined = array.combined();
        assert_eq!(
            (combined.count(1), combined.count(3), combined.count(7)),
            (5, 4, 1)
        );

        let mut doubled = array.clone();
        doubled += &array;
        assert_eq!(doubled.total(), 20);
        assert_eq!(doubled.get([0, 1]).unwrap().count(3), 8);
    }

    #[test]
    fn min_and_max_skip_cells_without_counts() {
        let mut array = CountDistributionArray::new([3]);
        assert_eq!(array.min(), None);
        assert_eq!(array.mean(), 0.0);
        assert_eq!(array.fraction_above(0), 0.0);
        array.add([2], 4);
        assert_eq!(array.min(), Some(4));
        assert_eq!(array.max(), Some(4));
    }

    #[test]
    #[should_panic(
        expected = "cannot add a count distribution array of shape [2] to one of shape [3]"
    )]
    fn adding_needs_the_same_shape() {
        let mut array = CountDistributionArray::new([3]);
        array += &CountDistributionArray::new([2]);
    }
}
