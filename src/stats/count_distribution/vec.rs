use core::fmt::Display;
use core::ops::AddAssign;
use std::io::{self, Write};

use super::{
    CountDistribution, Zeros, combined_of, max_of, min_of, ratio, sum_of, total_above_of, total_of,
};
use crate::stats::algorithms::weighted_correlation;

/// A [`CountDistribution`] per ID `0..len`; adding to an ID beyond grows it.
///
/// Summaries such as [`total`](Self::total), [`mean`](Self::mean) or [`max`](Self::max) are over the
/// values of all IDs.
///
/// ```
/// use coretools_rs::stats::CountDistributionVec;
///
/// let mut depth_per_sample = CountDistributionVec::new();
/// depth_per_sample.add(0, 3);
/// depth_per_sample.add(1, 5);
/// depth_per_sample.add(1, 5);
/// assert_eq!(depth_per_sample.get(1).unwrap().count(5), 2);
/// assert_eq!(depth_per_sample.combined().mode(), Some(5));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CountDistributionVec {
    distributions: Vec<CountDistribution>,
}

impl CountDistributionVec {
    pub const fn new() -> Self {
        Self {
            distributions: Vec::new(),
        }
    }

    /// The IDs `0..len`, without counts.
    pub fn with_len(len: usize) -> Self {
        Self {
            distributions: vec![CountDistribution::new(); len],
        }
    }

    /// Tracks the IDs `0..len`: drops larger IDs with their distributions, or adds IDs without counts.
    pub fn resize(&mut self, len: usize) {
        self.distributions.resize(len, CountDistribution::new());
    }

    /// [Resizes](CountDistribution::resize) the distribution of every ID to track the values `0..len`.
    pub fn resize_distributions(&mut self, len: usize) {
        for dist in &mut self.distributions {
            dist.resize(len);
        }
    }

    /// Forgets all IDs.
    pub fn clear(&mut self) {
        self.distributions.clear();
    }

    /// The number of IDs.
    pub fn len(&self) -> usize {
        self.distributions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.distributions.is_empty()
    }

    pub fn get(&self, id: usize) -> Option<&CountDistribution> {
        self.distributions.get(id)
    }

    pub fn iter(&self) -> core::slice::Iter<'_, CountDistribution> {
        self.distributions.iter()
    }

    /// The distribution of `id`, growing to track it if needed.
    fn distribution_mut(&mut self, id: usize) -> &mut CountDistribution {
        if id >= self.distributions.len() {
            self.distributions.resize(id + 1, CountDistribution::new());
        }
        &mut self.distributions[id]
    }

    /// Counts one occurrence of `value` for `id`.
    pub fn add(&mut self, id: usize, value: usize) {
        self.distribution_mut(id).add(value);
    }

    /// Counts `n` occurrences of `value` for `id`.
    pub fn add_n(&mut self, id: usize, value: usize, n: u64) {
        self.distribution_mut(id).add_n(value, n);
    }

    /// Adds the counts of `dist` to those of `id`.
    pub fn add_distribution(&mut self, id: usize, dist: &CountDistribution) {
        *self.distribution_mut(id) += dist;
    }

    /// The number of occurrences of all values of all IDs.
    pub fn total(&self) -> u64 {
        total_of(self.iter())
    }

    /// How often `value` occurred over all IDs.
    pub fn total_at(&self, value: usize) -> u64 {
        self.iter().map(|dist| dist.count(value)).sum()
    }

    /// The number of occurrences of values above `value` over all IDs.
    pub fn total_above(&self, value: usize) -> u64 {
        total_above_of(self.iter(), value)
    }

    /// The sum of all occurrences of all IDs.
    pub fn sum(&self) -> u64 {
        sum_of(self.iter())
    }

    /// The mean of all occurrences of all IDs; 0 without counts.
    pub fn mean(&self) -> f64 {
        ratio(self.sum(), self.total())
    }

    /// The fraction of occurrences above `value` over all IDs; 0 without counts.
    pub fn fraction_above(&self, value: usize) -> f64 {
        ratio(self.total_above(value), self.total())
    }

    /// The smallest value that occurred for any ID; `None` without counts.
    pub fn min(&self) -> Option<usize> {
        min_of(self.iter())
    }

    /// The largest value that occurred for any ID; `None` without counts.
    pub fn max(&self) -> Option<usize> {
        max_of(self.iter())
    }

    /// The distribution of the values of all IDs together.
    pub fn combined(&self) -> CountDistribution {
        combined_of(self.iter())
    }

    /// The squared Pearson correlation between ID and value over all occurrences; `None` without
    /// variation in IDs or values.
    pub fn r_squared(&self) -> Option<f64> {
        let occurrences = self.iter().enumerate().flat_map(|(id, dist)| {
            dist.iter()
                .map(move |(value, count)| (id as f64, value as f64, count as f64))
        });
        weighted_correlation(occurrences).map(|r| r * r)
    }

    /// Writes the row `id<TAB>value<TAB>count` per tracked value of every ID with counts.
    pub fn write<W: Write + ?Sized>(&self, out: &mut W, zeros: Zeros) -> io::Result<()> {
        for (id, dist) in self.iter().enumerate() {
            if dist.total() > 0 {
                dist.write_labelled(out, id, zeros)?;
            }
        }
        Ok(())
    }

    /// Writes the row `name<TAB>value<TAB>count` per tracked value of every ID with counts, where
    /// `names[id]` names the ID.
    ///
    /// # Panics
    ///
    /// If there is not exactly one name per ID.
    pub fn write_named<W: Write + ?Sized>(
        &self,
        out: &mut W,
        names: &[impl Display],
        zeros: Zeros,
    ) -> io::Result<()> {
        self.assert_one_name_per_id(names.len());
        for (name, dist) in names.iter().zip(self.iter()) {
            if dist.total() > 0 {
                dist.write_labelled(out, name, zeros)?;
            }
        }
        Ok(())
    }

    /// Writes a matrix with one row per ID and one column per value `0..=max` (just `0` without
    /// counts): the header `corner<TAB>0<TAB>1…`, then `id<TAB>count of 0<TAB>count of 1…` for
    /// every ID.
    pub fn write_matrix<W: Write + ?Sized>(
        &self,
        out: &mut W,
        corner: impl Display,
    ) -> io::Result<()> {
        let columns = self.max().unwrap_or(0) + 1;
        write_header(out, corner, 0..columns)?;
        write_rows(out, columns, self.iter().enumerate())
    }

    /// Like [`write_matrix`](Self::write_matrix), with the columns named `value_names` instead of
    /// `0, 1, …`.
    ///
    /// # Panics
    ///
    /// If a value that occurred has no name.
    pub fn write_matrix_named<W: Write + ?Sized>(
        &self,
        out: &mut W,
        corner: impl Display,
        value_names: &[impl Display],
    ) -> io::Result<()> {
        self.assert_name_per_value(value_names.len());
        write_header(out, corner, value_names)?;
        write_rows(out, value_names.len(), self.iter().enumerate())
    }

    /// Like [`write_matrix_named`](Self::write_matrix_named), with a first row `combined_name` for
    /// the [combined](Self::combined) distribution and the rows named `row_names` instead of by ID.
    ///
    /// # Panics
    ///
    /// If a value that occurred has no name, or if there is not exactly one row name per ID.
    pub fn write_matrix_with_combined<W: Write + ?Sized, N: Display>(
        &self,
        out: &mut W,
        corner: impl Display,
        value_names: &[impl Display],
        combined_name: N,
        row_names: &[N],
    ) -> io::Result<()> {
        self.assert_name_per_value(value_names.len());
        self.assert_one_name_per_id(row_names.len());
        write_header(out, corner, value_names)?;
        let combined = self.combined();
        write_rows(out, value_names.len(), [(combined_name, &combined)])?;
        write_rows(out, value_names.len(), row_names.iter().zip(self.iter()))
    }

    fn assert_one_name_per_id(&self, names: usize) {
        let ids = self.len();
        assert_eq!(names, ids, "{ids} IDs need {ids} names, not {names}");
    }

    fn assert_name_per_value(&self, names: usize) {
        if let Some(max) = self.max() {
            assert!(
                max < names,
                "values up to {max} need at least {} names, not {names}",
                max + 1
            );
        }
    }
}

impl AddAssign<&CountDistributionVec> for CountDistributionVec {
    /// Adds the distribution of every ID of `other` to the same ID, growing to its IDs if needed.
    fn add_assign(&mut self, other: &CountDistributionVec) {
        for (id, dist) in other.iter().enumerate() {
            self.add_distribution(id, dist);
        }
    }
}

impl<'a> IntoIterator for &'a CountDistributionVec {
    type Item = &'a CountDistribution;
    type IntoIter = core::slice::Iter<'a, CountDistribution>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

fn write_header<W: Write + ?Sized, C: Display>(
    out: &mut W,
    corner: impl Display,
    columns: impl IntoIterator<Item = C>,
) -> io::Result<()> {
    write!(out, "{corner}")?;
    for column in columns {
        write!(out, "\t{column}")?;
    }
    writeln!(out)
}

/// Writes `label<TAB>count of 0<TAB>…<TAB>count of columns - 1` per distribution.
fn write_rows<'a, W: Write + ?Sized, L: Display>(
    out: &mut W,
    columns: usize,
    rows: impl IntoIterator<Item = (L, &'a CountDistribution)>,
) -> io::Result<()> {
    for (label, dist) in rows {
        write!(out, "{label}")?;
        for value in 0..columns {
            write!(out, "\t{}", dist.count(value))?;
        }
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_float_eq;

    fn complicated() -> CountDistributionVec {
        let mut vec = CountDistributionVec::with_len(10);
        vec.add_n(0, 0, 10);
        vec.add_n(0, 0, 1);
        vec.add_n(0, 1, 100);
        vec.add_n(10, 0, 400);
        vec.add_n(10, 1, 800);
        vec.add_n(10, 5, 200);
        vec
    }

    /// The distributions of ID 0 (11 × 0, 100 × 1) and 10 (200 × 5) and nothing in between.
    #[track_caller]
    fn assert_ids_0_and_10(vec: &CountDistributionVec, count_of_id_10: u64) {
        assert_eq!(vec.len(), 11);
        let id_0 = vec.get(0).unwrap();
        assert_eq!((id_0.count(0), id_0.count(1)), (11, 100));
        assert!((1..10).all(|id| vec.get(id).unwrap().total() == 0));
        let id_10 = vec.get(10).unwrap();
        assert!((0..5).all(|value| id_10.count(value) == 0));
        assert_eq!(id_10.count(5), count_of_id_10);
    }

    fn written(write: impl FnOnce(&mut Vec<u8>) -> io::Result<()>) -> String {
        let mut out = Vec::new();
        write(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn with_len_has_no_counts() {
        let vec = CountDistributionVec::with_len(10);
        assert_eq!(vec.len(), 10);
        assert!(vec.iter().all(|dist| dist.is_empty()));
    }

    #[test]
    fn adding_grows_to_the_largest_id() {
        let mut vec = CountDistributionVec::with_len(5);
        for _ in 0..11 {
            vec.add(0, 0);
        }
        for _ in 0..100 {
            vec.add(0, 1);
        }
        vec.add(10, 5);
        assert_ids_0_and_10(&vec, 1);
    }

    #[test]
    fn adding_several_counts_at_once() {
        let mut vec = CountDistributionVec::with_len(5);
        vec.add_n(0, 0, 10);
        vec.add_n(0, 0, 1);
        vec.add_n(0, 1, 100);
        vec.add_n(10, 5, 200);
        assert_ids_0_and_10(&vec, 200);

        let mut merged = CountDistributionVec::new();
        merged += &vec;
        assert_ids_0_and_10(&merged, 200);
    }

    #[test]
    fn adding_a_distribution_to_an_id() {
        let mut dist = CountDistribution::with_len(5);
        dist.add_n(0, 10);
        dist.add_n(0, 1);
        dist.add_n(1, 100);

        let mut vec = CountDistributionVec::new();
        vec.add_distribution(0, &dist);
        vec.add_n(10, 5, 200);
        assert_ids_0_and_10(&vec, 200);
    }

    #[test]
    fn clear_forgets_all_ids() {
        let mut vec = CountDistributionVec::with_len(10);
        vec.clear();
        assert!(vec.is_empty());
        assert!(vec.get(0).is_none());
    }

    #[test]
    fn summaries_over_all_ids() {
        let vec = complicated();
        assert_eq!(vec.total(), 1511);
        assert_eq!(vec.total_above(1), 200);
        assert_eq!(vec.total_above(0), 1100);
        assert_eq!(vec.sum(), 1900);
        assert_float_eq(vec.mean(), 1.257_445);
        assert_eq!(vec.min(), Some(0));
        assert_eq!(vec.max(), Some(5));
        assert_float_eq(vec.fraction_above(1), 0.132_362_7);
        assert_float_eq(vec.fraction_above(0), 0.727_994_7);
        assert!(vec.get(10).is_some());
        assert!(vec.get(11).is_none());
    }

    #[test]
    fn min_and_max_skip_ids_without_counts() {
        let mut vec = CountDistributionVec::with_len(3);
        vec.add(5, 3);
        vec.add(6, 8);
        assert_eq!(vec.min(), Some(3));
        assert_eq!(vec.max(), Some(8));
    }

    #[test]
    fn no_counts_have_zero_mean_and_fractions() {
        let vec = CountDistributionVec::with_len(3);
        assert_eq!(vec.min(), None);
        assert_eq!(vec.max(), None);
        assert_eq!(vec.mean(), 0.0);
        assert_eq!(vec.fraction_above(0), 0.0);
    }

    #[test]
    fn combined_distribution_of_all_ids() {
        let combined = complicated().combined();
        assert_eq!(
            combined.iter().map(|(_, count)| count).collect::<Vec<_>>(),
            [411, 900, 0, 0, 0, 200]
        );
        assert_eq!(complicated().total_at(1), 900);
    }

    #[test]
    fn r_squared_between_id_and_value() {
        let mut vec = CountDistributionVec::new();
        vec.add_n(0, 0, 10);
        vec.add_n(0, 1, 20);
        vec.add_n(0, 2, 30);
        vec.add_n(1, 0, 40);
        vec.add_n(1, 1, 50);
        vec.add_n(1, 2, 60);
        vec.add_n(1, 3, 70);
        assert_float_eq(vec.r_squared().unwrap(), 0.024_043_25);
    }

    #[test]
    fn r_squared_needs_variation_in_id_and_value() {
        let mut vec = CountDistributionVec::new();
        assert_eq!(vec.r_squared(), None);
        vec.add_n(0, 1, 10);
        vec.add_n(0, 2, 10);
        assert_eq!(vec.r_squared(), None);
    }

    #[test]
    fn writes_the_combined_distribution() {
        let combined = complicated().combined();
        assert_eq!(
            written(|out| combined.write(out, Zeros::Include)),
            "0\t411\n1\t900\n2\t0\n3\t0\n4\t0\n5\t200\n"
        );
        assert_eq!(
            written(|out| combined.write_labelled(out, "ID", Zeros::Skip)),
            "ID\t0\t411\nID\t1\t900\nID\t5\t200\n"
        );
    }

    #[test]
    fn writes_one_row_per_id_and_value() {
        let vec = complicated();
        let rows = "0\t0\t11\n0\t1\t100\n\
                    10\t0\t400\n10\t1\t800\n10\t2\t0\n10\t3\t0\n10\t4\t0\n10\t5\t200\n";
        assert_eq!(written(|out| vec.write(out, Zeros::Include)), rows);
        assert_eq!(
            written(|out| vec.write(out, Zeros::Skip)),
            "0\t0\t11\n0\t1\t100\n10\t0\t400\n10\t1\t800\n10\t5\t200\n"
        );

        let names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"];
        assert_eq!(
            written(|out| vec.write_named(out, &names, Zeros::Include)),
            "a\t0\t11\na\t1\t100\n\
             k\t0\t400\nk\t1\t800\nk\t2\t0\nk\t3\t0\nk\t4\t0\nk\t5\t200\n"
        );
    }

    #[test]
    #[should_panic(expected = "11 IDs need 11 names, not 10")]
    fn writing_needs_a_name_per_id() {
        let names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
        complicated()
            .write_named(&mut Vec::new(), &names, Zeros::Include)
            .unwrap();
    }

    #[test]
    fn writes_a_matrix_of_ids_by_values() {
        let vec = complicated();
        let rows: String = (1..10)
            .map(|id| format!("{id}\t0\t0\t0\t0\t0\t0\n"))
            .collect();
        let body = format!("0\t11\t100\t0\t0\t0\t0\n{rows}10\t400\t800\t0\t0\t0\t200\n");
        assert_eq!(
            written(|out| vec.write_matrix(out, "ID/label")),
            format!("ID/label\t0\t1\t2\t3\t4\t5\n{body}")
        );
        assert_eq!(
            written(|out| vec.write_matrix_named(out, "ID", &["a", "b", "c", "d", "e", "f"])),
            format!("ID\ta\tb\tc\td\te\tf\n{body}")
        );
    }

    #[test]
    fn writes_a_matrix_with_the_combined_distribution_first() {
        let mut vec = CountDistributionVec::new();
        vec.add_n(0, 0, 1);
        vec.add_n(1, 1, 2);
        assert_eq!(
            written(|out| vec.write_matrix_with_combined(
                out,
                "ID",
                &["x", "y"],
                "all",
                &["p", "q"]
            )),
            "ID\tx\ty\nall\t1\t2\np\t1\t0\nq\t0\t2\n"
        );
    }

    #[test]
    fn a_matrix_without_counts_has_the_column_of_value_0() {
        let vec = CountDistributionVec::with_len(2);
        assert_eq!(
            written(|out| vec.write_matrix(out, "ID")),
            "ID\t0\n0\t0\n1\t0\n"
        );
    }

    #[test]
    #[should_panic(expected = "2 IDs need 2 names, not 3")]
    fn a_combined_matrix_needs_a_row_name_per_id() {
        let mut vec = CountDistributionVec::new();
        vec.add_n(1, 1, 2);
        vec.write_matrix_with_combined(
            &mut Vec::new(),
            "ID",
            &["x", "y"],
            "all",
            &["all", "p", "q"],
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "values up to 5 need at least 6 names, not 5")]
    fn a_matrix_needs_a_name_per_value() {
        complicated()
            .write_matrix_named(&mut Vec::new(), "ID", &["a", "b", "c", "d", "e"])
            .unwrap();
    }
}
