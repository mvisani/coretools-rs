//! Count distributions: histograms of how often each value occurred.

mod array;
mod map;
mod vec;

pub use array::CountDistributionArray;
pub use map::CountDistributionMap;
pub use vec::CountDistributionVec;

use core::fmt::Display;
use core::ops::AddAssign;
use std::io::{self, Write};

/// Whether a written count distribution has rows for values without counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zeros {
    Include,
    Skip,
}

impl Zeros {
    fn writes(self, count: u64) -> bool {
        self == Self::Include || count > 0
    }
}

/// How often each value `0, 1, 2, …` occurred: the values 1 7 4 1 2 7 7 2 have the counts
/// 0 2 2 0 1 0 0 3.
///
/// The distribution tracks the values `0..len`; adding a larger value grows it. Distributions of
/// disjoint sets of values combine with `+=`.
///
/// ```
/// use coretools_rs::stats::CountDistribution;
///
/// let depths: CountDistribution = [1, 7, 4, 1, 2, 7, 7, 2].into_iter().collect();
/// assert_eq!(depths.count(7), 3);
/// assert_eq!(depths.mode(), Some(7));
/// assert_eq!(depths.mean(), 3.875);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CountDistribution {
    counts: Vec<u64>,
}

impl CountDistribution {
    pub const fn new() -> Self {
        Self { counts: Vec::new() }
    }

    /// The values `0..len`, without counts.
    pub fn with_len(len: usize) -> Self {
        Self {
            counts: vec![0; len],
        }
    }

    /// Tracks the values `0..len`: drops larger values with their counts, or adds values without counts.
    pub fn resize(&mut self, len: usize) {
        self.counts.resize(len, 0);
    }

    /// Forgets all values.
    pub fn clear(&mut self) {
        self.counts.clear();
    }

    /// Sets every count to 0, still tracking the values `0..len`.
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }

    /// The number of tracked values: one more than the largest value added or resized to.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether no values are tracked; a distribution of tracked values without counts is not empty
    /// (whether there are counts is `total() > 0`).
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// How often `value` occurred; 0 beyond the tracked values.
    pub fn count(&self, value: usize) -> u64 {
        self.counts.get(value).copied().unwrap_or(0)
    }

    /// The count of `value`, growing to track it if needed.
    pub fn count_mut(&mut self, value: usize) -> &mut u64 {
        if value >= self.counts.len() {
            self.counts.resize(value + 1, 0);
        }
        &mut self.counts[value]
    }

    /// Counts one occurrence of `value`.
    pub fn add(&mut self, value: usize) {
        self.add_n(value, 1);
    }

    /// Counts `n` occurrences of `value`.
    pub fn add_n(&mut self, value: usize, n: u64) {
        *self.count_mut(value) += n;
    }

    /// `(value, count)` for every tracked value, in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u64)> + Clone + '_ {
        self.counts.iter().copied().enumerate()
    }

    /// The number of occurrences of all values.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// The number of occurrences of values above `value`.
    pub fn total_above(&self, value: usize) -> u64 {
        self.counts.iter().skip(value.saturating_add(1)).sum()
    }

    /// The sum of all occurrences: `Σ value × count`.
    pub fn sum(&self) -> u64 {
        self.iter().map(|(value, count)| value as u64 * count).sum()
    }

    /// The mean of all occurrences; 0 without counts.
    pub fn mean(&self) -> f64 {
        ratio(self.sum(), self.total())
    }

    /// The fraction of occurrences that are `value`; 0 without counts.
    pub fn fraction(&self, value: usize) -> f64 {
        ratio(self.count(value), self.total())
    }

    /// The fraction of occurrences above `value`; 0 without counts.
    pub fn fraction_above(&self, value: usize) -> f64 {
        ratio(self.total_above(value), self.total())
    }

    /// The most frequent value, the smallest one among ties; `None` without counts.
    pub fn mode(&self) -> Option<usize> {
        mode_of(&self.counts, 0)
    }

    /// The most frequent value ignoring the `cutoff` smallest and the `cutoff` largest tracked values,
    /// the smallest one among ties; `None` without counts in between.
    pub fn mode_trimmed(&self, cutoff: usize) -> Option<usize> {
        let end = self.counts.len().saturating_sub(cutoff);
        self.counts
            .get(cutoff..end)
            .and_then(|trimmed| mode_of(trimmed, cutoff))
    }

    /// The smallest value whose cumulative count exceeds half of the total (the upper median for an
    /// even total); `None` without counts.
    pub fn median(&self) -> Option<usize> {
        let half = self.total() / 2;
        let mut cumulative = 0;
        self.counts.iter().position(|&count| {
            cumulative += count;
            cumulative > half
        })
    }

    /// The smallest value that occurred; `None` without counts.
    pub fn min(&self) -> Option<usize> {
        self.counts.iter().position(|&count| count > 0)
    }

    /// The largest value that occurred; `None` without counts.
    pub fn max(&self) -> Option<usize> {
        self.counts.iter().rposition(|&count| count > 0)
    }

    /// Writes the row `value<TAB>count` per tracked value.
    pub fn write<W: Write + ?Sized>(&self, out: &mut W, zeros: Zeros) -> io::Result<()> {
        for (value, count) in self.iter().filter(|&(_, count)| zeros.writes(count)) {
            writeln!(out, "{value}\t{count}")?;
        }
        Ok(())
    }

    /// Writes the row `label<TAB>value<TAB>count` per tracked value.
    pub fn write_labelled<W: Write + ?Sized>(
        &self,
        out: &mut W,
        label: impl Display,
        zeros: Zeros,
    ) -> io::Result<()> {
        for (value, count) in self.iter().filter(|&(_, count)| zeros.writes(count)) {
            writeln!(out, "{label}\t{value}\t{count}")?;
        }
        Ok(())
    }
}

impl AddAssign<&CountDistribution> for CountDistribution {
    fn add_assign(&mut self, other: &CountDistribution) {
        if other.counts.len() > self.counts.len() {
            self.counts.resize(other.counts.len(), 0);
        }
        for (count, other) in self.counts.iter_mut().zip(&other.counts) {
            *count += other;
        }
    }
}

impl Extend<usize> for CountDistribution {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, values: I) {
        values.into_iter().for_each(|value| self.add(value));
    }
}

impl FromIterator<usize> for CountDistribution {
    fn from_iter<I: IntoIterator<Item = usize>>(values: I) -> Self {
        let mut dist = Self::new();
        dist.extend(values);
        dist
    }
}

/// `numerator / denominator`, 0 for a denominator of 0.
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        // counts beyond 2^53 round, as any f64 does
        numerator as f64 / denominator as f64
    }
}

/// The position of the first largest count plus `offset`; `None` if all counts are 0.
fn mode_of(counts: &[u64], offset: usize) -> Option<usize> {
    let mut mode = None;
    let mut largest = 0;
    for (value, &count) in counts.iter().enumerate() {
        if count > largest {
            largest = count;
            mode = Some(offset + value);
        }
    }
    mode
}

// Summaries of several distributions taken together.

fn total_of<'a>(dists: impl Iterator<Item = &'a CountDistribution>) -> u64 {
    dists.map(CountDistribution::total).sum()
}

fn total_above_of<'a>(dists: impl Iterator<Item = &'a CountDistribution>, value: usize) -> u64 {
    dists.map(|dist| dist.total_above(value)).sum()
}

fn sum_of<'a>(dists: impl Iterator<Item = &'a CountDistribution>) -> u64 {
    dists.map(CountDistribution::sum).sum()
}

fn min_of<'a>(dists: impl Iterator<Item = &'a CountDistribution>) -> Option<usize> {
    dists.filter_map(CountDistribution::min).min()
}

fn max_of<'a>(dists: impl Iterator<Item = &'a CountDistribution>) -> Option<usize> {
    dists.filter_map(CountDistribution::max).max()
}

fn combined_of<'a>(dists: impl Iterator<Item = &'a CountDistribution>) -> CountDistribution {
    let mut combined = CountDistribution::new();
    for dist in dists {
        combined += dist;
    }
    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_float_eq;

    /// 100 × 0, 1 × 1, 10 × 2, 200 × 10.
    fn skewed() -> CountDistribution {
        let mut dist = CountDistribution::new();
        dist.add_n(0, 100);
        dist.add(1);
        dist.add_n(2, 10);
        dist.add_n(10, 200);
        dist
    }

    fn written(write: impl FnOnce(&mut Vec<u8>) -> io::Result<()>) -> String {
        let mut out = Vec::new();
        write(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn with_len_tracks_values_without_counts() {
        let dist = CountDistribution::with_len(10);
        assert_eq!(dist.len(), 10);
        assert!(!dist.is_empty());
        assert!((0..10).all(|value| dist.count(value) == 0));
        assert_eq!(dist.total(), 0);
    }

    #[test]
    fn adding_grows_to_the_largest_value() {
        let mut dist = CountDistribution::new();
        dist.add(0);
        dist.add(0);
        dist.add(1);
        dist.add(10);
        assert_eq!(dist.len(), 11);
        assert_eq!(dist.count(0), 2);
        assert_eq!(dist.count(1), 1);
        assert_eq!(dist.count(10), 1);
        assert!((2..10).all(|value| dist.count(value) == 0));
    }

    #[test]
    fn adding_several_counts_at_once() {
        let mut dist = CountDistribution::new();
        dist.add_n(0, 10);
        dist.add_n(0, 1);
        dist.add_n(1, 100);
        dist.add_n(10, 200);
        assert_eq!(dist.count(0), 11);
        assert_eq!(dist.count(1), 100);
        assert_eq!(dist.count(10), 200);
        assert!((2..10).all(|value| dist.count(value) == 0));

        let mut merged = CountDistribution::new();
        merged += &dist;
        assert_eq!(merged, dist);
        merged += &dist;
        assert_eq!(merged.count(0), 22);
        assert_eq!(merged.count(10), 400);
    }

    #[test]
    fn counts_are_mutable_in_place_and_grow() {
        let mut dist = skewed();
        *dist.count_mut(0) += 10;
        assert_eq!(dist.count(0), 110);
        *dist.count_mut(20) = 150;
        assert_eq!(dist.count(20), 150);
        assert_eq!(dist.len(), 21);
    }

    #[test]
    fn values_beyond_the_largest_have_no_counts() {
        assert_eq!(CountDistribution::new().count(100), 0);
        assert_eq!(skewed().count(100), 0);
    }

    #[test]
    fn clear_forgets_all_values() {
        let mut dist = CountDistribution::new();
        assert!(dist.is_empty());
        dist.add(10);
        assert!(!dist.is_empty());
        dist.clear();
        assert!(dist.is_empty());
        assert_eq!(dist.total(), 0);
    }

    #[test]
    fn reset_forgets_the_counts_but_not_the_values() {
        let mut dist: CountDistribution = [1, 3, 3].into_iter().collect();
        dist.reset();
        assert_eq!(dist, CountDistribution::with_len(4));
    }

    #[test]
    fn summaries() {
        let dist = skewed();
        assert_eq!(dist.total(), 311);
        assert_eq!(dist.total_above(1), 210);
        assert_eq!(dist.total_above(0), 211);
        assert_eq!(dist.sum(), 2021);
        assert_float_eq(dist.mean(), 6.498_392);
        assert_float_eq(dist.fraction_above(1), 0.675_241_2);
        assert_float_eq(dist.fraction_above(0), 0.678_456_6);
        assert_float_eq(dist.fraction(10), 200.0 / 311.0);
        assert_eq!(dist.mode(), Some(10));
        assert_eq!(dist.min(), Some(0));
        assert_eq!(dist.max(), Some(10));
    }

    #[test]
    fn mode_ties_go_to_the_smallest_value() {
        let mut dist = CountDistribution::new();
        dist.add_n(0, 100);
        dist.add_n(1, 100);
        assert_eq!(dist.mode(), Some(0));
    }

    #[test]
    fn min_and_max_skip_values_without_counts() {
        let mut dist = CountDistribution::new();
        dist.add_n(2, 10);
        dist.add_n(10, 200);
        dist.add_n(20, 0);
        assert_eq!(dist.min(), Some(2));
        assert_eq!(dist.max(), Some(10));
    }

    #[test]
    fn median_is_the_value_where_the_cumulative_count_passes_half() {
        let mut dist = CountDistribution::new();
        for value in 0..100 {
            dist.add_n(value, value as u64);
        }
        assert_eq!(dist.median(), Some(70));
    }

    #[test]
    fn mode_trimmed_ignores_the_values_at_both_ends() {
        let mut dist = CountDistribution::new();
        for (value, count) in [5, 1, 2, 9].into_iter().enumerate() {
            dist.add_n(value, count);
        }
        assert_eq!(dist.mode(), Some(3));
        assert_eq!(dist.mode_trimmed(1), Some(2));
        assert_eq!(dist.mode_trimmed(2), None);
        assert_eq!(dist.mode_trimmed(3), None);
    }

    #[test]
    fn a_distribution_without_counts_has_no_mode_median_min_or_max() {
        for dist in [CountDistribution::new(), CountDistribution::with_len(5)] {
            assert_eq!(dist.mode(), None);
            assert_eq!(dist.median(), None);
            assert_eq!(dist.min(), None);
            assert_eq!(dist.max(), None);
            assert_eq!(dist.mean(), 0.0);
            assert_eq!(dist.fraction(0), 0.0);
            assert_eq!(dist.fraction_above(0), 0.0);
        }
    }

    #[test]
    fn iterates_over_every_value_up_to_the_largest() {
        let mut dist = CountDistribution::new();
        dist.add_n(1, 3);
        dist.add_n(3, 1);
        assert_eq!(
            dist.iter().collect::<Vec<_>>(),
            [(0, 0), (1, 3), (2, 0), (3, 1)]
        );
    }

    #[test]
    fn writes_one_row_per_value() {
        let mut dist = CountDistribution::new();
        for value in 0..4 {
            dist.add_n(value, (value * value) as u64);
        }
        assert_eq!(
            written(|out| dist.write(out, Zeros::Include)),
            "0\t0\n1\t1\n2\t4\n3\t9\n"
        );
        assert_eq!(
            written(|out| dist.write(out, Zeros::Skip)),
            "1\t1\n2\t4\n3\t9\n"
        );
        assert_eq!(
            written(|out| dist.write_labelled(out, "ID", Zeros::Include)),
            "ID\t0\t0\nID\t1\t1\nID\t2\t4\nID\t3\t9\n"
        );
    }
}
