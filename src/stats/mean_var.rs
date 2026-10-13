//! Streaming count, sum, mean and variance.

use core::borrow::Borrow;
use core::ops::{Add, AddAssign};
use std::collections::BTreeMap;

/// Count, sum, mean, population variance and standard deviation of the values added so far,
/// accumulated in one pass with Welford's algorithm: numerically stable, also for values far from 0.
///
/// Accumulators of disjoint sets of values combine with `+`. Without values, every moment is 0.
///
/// ```
/// use coretools_rs::stats::MeanVar;
///
/// let moments: MeanVar = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0].into_iter().collect();
/// assert_eq!(moments.mean(), 5.0);
/// assert_eq!(moments.variance(), 4.0);
/// assert_eq!(moments.sd(), 2.0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MeanVar {
    count: u64,
    sum: f64,
    mean: f64,
    /// Sum of squared deviations from the mean.
    squared_deviations: f64,
}

impl MeanVar {
    pub const fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            mean: 0.0,
            squared_deviations: 0.0,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        let deviation = value - self.mean;
        // u64 → f64 rounds only beyond 2^53 values
        self.mean += deviation / self.count as f64;
        self.squared_deviations += deviation * (value - self.mean);
    }

    /// The number of values.
    pub const fn count(&self) -> u64 {
        self.count
    }

    pub const fn sum(&self) -> f64 {
        self.sum
    }

    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// The population variance `Σ (xᵢ - x̄)² / n` (not the sample variance, which divides by `n - 1`).
    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.squared_deviations / self.count as f64
        }
    }

    /// The population standard deviation.
    pub fn sd(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl AddAssign for MeanVar {
    /// Chan et al.'s pairwise update.
    fn add_assign(&mut self, other: Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other;
            return;
        }
        let count = self.count + other.count;
        let (n_self, n_other, n) = (self.count as f64, other.count as f64, count as f64);
        let deviation = other.mean - self.mean;
        self.count = count;
        self.sum += other.sum;
        self.mean += deviation * n_other / n;
        self.squared_deviations +=
            other.squared_deviations + deviation * deviation * n_self * n_other / n;
    }
}

impl Add for MeanVar {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl Extend<f64> for MeanVar {
    fn extend<I: IntoIterator<Item = f64>>(&mut self, values: I) {
        values.into_iter().for_each(|v| self.push(v));
    }
}

impl FromIterator<f64> for MeanVar {
    fn from_iter<I: IntoIterator<Item = f64>>(values: I) -> Self {
        let mut moments = Self::new();
        moments.extend(values);
        moments
    }
}

/// A [`MeanVar`] per key, ordered by key.
///
/// ```
/// use coretools_rs::stats::MeanVarMap;
///
/// let mut by_sample = MeanVarMap::new();
/// by_sample.add("a", 1.0);
/// by_sample.add("a", 3.0);
/// by_sample.add("b", 10.0);
/// assert_eq!(by_sample.get("a").unwrap().mean(), 2.0);
/// assert_eq!(by_sample.sum(), 14.0);
/// ```
#[derive(Debug, Clone)]
pub struct MeanVarMap<K> {
    moments: BTreeMap<K, MeanVar>,
}

impl<K: Ord> MeanVarMap<K> {
    pub const fn new() -> Self {
        Self {
            moments: BTreeMap::new(),
        }
    }

    /// Adds `value` to the moments of `key`.
    pub fn add(&mut self, key: K, value: f64) {
        self.moments.entry(key).or_default().push(value);
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&MeanVar>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.moments.get(key)
    }
}

impl<K> MeanVarMap<K> {
    /// The number of keys.
    pub fn len(&self) -> usize {
        self.moments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moments.is_empty()
    }

    /// The sum of all values of all keys.
    pub fn sum(&self) -> f64 {
        self.moments.values().map(MeanVar::sum).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &MeanVar)> {
        self.moments.iter()
    }

    pub fn clear(&mut self) {
        self.moments.clear();
    }
}

impl<K: Ord> Default for MeanVarMap<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`MeanVar`] per index `0..len`; adding to an index beyond grows it.
///
/// ```
/// use coretools_rs::stats::MeanVarVec;
///
/// let mut by_position = MeanVarVec::new();
/// by_position.add(2, 1.5);
/// assert_eq!(by_position.len(), 3);
/// assert_eq!(by_position.get(0).unwrap().count(), 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MeanVarVec {
    moments: Vec<MeanVar>,
}

impl MeanVarVec {
    pub const fn new() -> Self {
        Self {
            moments: Vec::new(),
        }
    }

    /// Indices `0..len` without values.
    pub fn with_len(len: usize) -> Self {
        Self {
            moments: vec![MeanVar::new(); len],
        }
    }

    /// Adds `value` to the moments of `index`, growing to `index + 1` indices if needed.
    pub fn add(&mut self, index: usize, value: f64) {
        if index >= self.moments.len() {
            self.moments.resize(index + 1, MeanVar::new());
        }
        self.moments[index].push(value);
    }

    pub fn get(&self, index: usize) -> Option<&MeanVar> {
        self.moments.get(index)
    }

    /// The number of indices.
    pub fn len(&self) -> usize {
        self.moments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moments.is_empty()
    }

    /// The sum of all values of all indices.
    pub fn sum(&self) -> f64 {
        self.moments.iter().map(MeanVar::sum).sum()
    }

    pub fn iter(&self) -> core::slice::Iter<'_, MeanVar> {
        self.moments.iter()
    }

    pub fn clear(&mut self) {
        self.moments.clear();
    }
}

impl<'a> IntoIterator for &'a MeanVarVec {
    type Item = &'a MeanVar;
    type IntoIter = core::slice::Iter<'a, MeanVar>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, assert_float_eq};

    const VALUES: [f64; 6] = [1.0, 2.0, 0.83, -102.9, 0.0, 100.293];

    #[track_caller]
    fn assert_moments_of_values(moments: &MeanVar) {
        assert_eq!(moments.count(), 6);
        assert_float_eq(moments.sum(), 1.223);
        assert_float_eq(moments.mean(), 0.203_833_3);
        assert_float_eq(moments.sd(), 58.669_32);
        assert_float_eq(moments.variance(), 3442.089);
    }

    #[test]
    fn moments_of_added_values() {
        let mut moments = MeanVar::new();
        for value in VALUES {
            moments.push(value);
        }
        assert_moments_of_values(&moments);
        assert_moments_of_values(&VALUES.into_iter().collect());
    }

    #[test]
    fn no_values_have_zero_moments() {
        let moments = MeanVar::new();
        assert_eq!(moments.count(), 0);
        assert_eq!(moments.sum(), 0.0);
        assert_eq!(moments.mean(), 0.0);
        assert_eq!(moments.variance(), 0.0);
        assert_eq!(moments.sd(), 0.0);
    }

    #[test]
    fn merging_gives_the_moments_of_all_values() {
        let mut merged = MeanVar::new();
        merged += VALUES.into_iter().collect();
        assert_moments_of_values(&merged);

        let (first, second) = VALUES.split_at(2);
        let first: MeanVar = first.iter().copied().collect();
        let second: MeanVar = second.iter().copied().collect();
        assert_moments_of_values(&(first + second));
        assert_moments_of_values(&(first + second + MeanVar::new()));
    }

    #[test]
    fn variance_is_exact_at_large_offsets() {
        let moments: MeanVar = [4.0, 7.0, 13.0, 16.0]
            .map(|v| 1e9 + v)
            .into_iter()
            .collect();
        assert_close(moments.mean(), 1e9 + 10.0);
        assert_close(moments.variance(), 22.5);

        let first: MeanVar = [1e9 + 4.0, 1e9 + 7.0].into_iter().collect();
        let second: MeanVar = [1e9 + 13.0, 1e9 + 16.0].into_iter().collect();
        assert_close((first + second).variance(), 22.5);
    }

    #[test]
    fn constant_values_have_zero_variance() {
        let moments: MeanVar = [0.1; 7].into_iter().collect();
        assert_eq!(moments.variance(), 0.0);
        assert_eq!(moments.sd(), 0.0);
    }

    #[test]
    fn accumulators_of_different_values_with_the_same_sum_differ() {
        let spread: MeanVar = [1.0, 3.0].into_iter().collect();
        let constant: MeanVar = [2.0, 2.0].into_iter().collect();
        assert_ne!(spread, constant);
        assert_eq!(spread, [1.0, 3.0].into_iter().collect());
    }

    #[test]
    fn squares_beyond_f64_do_not_overflow() {
        let root_of_max = f64::MAX.sqrt();
        let moments: MeanVar = [root_of_max, root_of_max].into_iter().collect();
        assert_eq!(moments.mean(), root_of_max);
        assert_eq!(moments.variance(), 0.0);
    }

    #[test]
    fn keyed_moments() {
        let mut map = MeanVarMap::new();
        for (key, value) in [0, 0, 0, 2, 2, 2].into_iter().zip(VALUES) {
            map.add(key, value);
        }
        assert_eq!(map.len(), 2);
        assert_float_eq(map.sum(), 1.223);
        assert_float_eq(map.get(&0).unwrap().mean(), 1.276_667);
        assert_float_eq(map.get(&2).unwrap().mean(), -0.869);
        assert!(map.get(&1).is_none());
        assert!(map.get(&3).is_none());
        let keys: Vec<_> = map.iter().map(|(key, _)| *key).collect();
        assert_eq!(keys, [0, 2]);

        map.clear();
        assert!(map.is_empty());
        assert!(map.get(&0).is_none());
    }

    #[test]
    fn indexed_moments_grow_to_the_largest_index() {
        let mut vec = MeanVarVec::new();
        for (index, value) in [0, 0, 0, 2, 2, 2].into_iter().zip(VALUES) {
            vec.add(index, value);
        }
        assert_eq!(vec.len(), 3);
        assert_float_eq(vec.sum(), 1.223);
        assert_float_eq(vec.get(0).unwrap().mean(), 1.276_667);
        assert_eq!(vec.get(1).unwrap().mean(), 0.0);
        assert_float_eq(vec.get(2).unwrap().mean(), -0.869);
        assert!(vec.get(3).is_none());

        vec.clear();
        assert!(vec.is_empty());
        assert!(vec.get(0).is_none());

        let vec = MeanVarVec::with_len(4);
        assert_eq!(vec.len(), 4);
        assert!(vec.iter().all(|moments| moments.count() == 0));
    }

    #[test]
    fn sums_of_integer_values_do_not_overflow_their_type() {
        // 18 × 15 exceeds u8::MAX
        let mut vec = MeanVarVec::new();
        let mut map = MeanVarMap::new();
        for index in 0..18_u8 {
            vec.add(usize::from(index), f64::from(15_u8));
            map.add(index, f64::from(15_u8));
        }
        assert_eq!(vec.sum(), 270.0);
        assert_eq!(map.sum(), 270.0);
    }
}
