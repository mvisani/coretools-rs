use core::borrow::Borrow;
use std::collections::BTreeMap;

use super::{CountDistribution, ratio, sum_of, total_of};

/// A [`CountDistribution`] per key, ordered by key.
///
/// ```
/// use coretools_rs::stats::CountDistributionMap;
///
/// let mut depth_per_chromosome = CountDistributionMap::new();
/// depth_per_chromosome.add("chr1", 3);
/// depth_per_chromosome.add("chr2", 5);
/// assert_eq!(depth_per_chromosome.get("chr2").unwrap().count(5), 1);
/// assert_eq!(depth_per_chromosome.mean(), 4.0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountDistributionMap<K> {
    distributions: BTreeMap<K, CountDistribution>,
}

impl<K: Ord> CountDistributionMap<K> {
    pub const fn new() -> Self {
        Self {
            distributions: BTreeMap::new(),
        }
    }

    /// Counts one occurrence of `value` for `key`.
    pub fn add(&mut self, key: K, value: usize) {
        self.distributions.entry(key).or_default().add(value);
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&CountDistribution>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.distributions.get(key)
    }
}

impl<K> CountDistributionMap<K> {
    /// The number of keys.
    pub fn len(&self) -> usize {
        self.distributions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.distributions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &CountDistribution)> {
        self.distributions.iter()
    }

    /// Forgets all keys.
    pub fn clear(&mut self) {
        self.distributions.clear();
    }

    /// The number of occurrences of all values of all keys.
    pub fn total(&self) -> u64 {
        total_of(self.distributions.values())
    }

    /// The sum of all occurrences of all keys.
    pub fn sum(&self) -> u64 {
        sum_of(self.distributions.values())
    }

    /// The mean of all occurrences of all keys; 0 without counts.
    pub fn mean(&self) -> f64 {
        ratio(self.sum(), self.total())
    }
}

impl<K: Ord> Default for CountDistributionMap<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_ids() -> CountDistributionMap<u64> {
        let mut map = CountDistributionMap::new();
        map.add(0, 0);
        map.add(0, 1);
        map.add(0, 1);
        map.add(10, 5);
        map
    }

    #[test]
    fn adds_to_the_distribution_of_each_key() {
        let map = two_ids();
        let id_0 = map.get(&0).unwrap();
        assert_eq!((id_0.count(0), id_0.count(1)), (1, 2));
        let id_10 = map.get(&10).unwrap();
        assert!((0..5).all(|value| id_10.count(value) == 0));
        assert_eq!(id_10.count(5), 1);
        assert!(map.get(&1).is_none());
        assert_eq!(map.iter().map(|(key, _)| *key).collect::<Vec<_>>(), [0, 10]);
    }

    #[test]
    fn summaries_over_all_keys() {
        let map = two_ids();
        assert_eq!(map.len(), 2);
        assert_eq!(map.total(), 4);
        assert_eq!(map.sum(), 7);
        assert_eq!(map.mean(), 1.75);
    }

    #[test]
    fn no_keys_have_zero_mean() {
        let mut map = two_ids();
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.total(), 0);
        assert_eq!(map.mean(), 0.0);
    }
}
