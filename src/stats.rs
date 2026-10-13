//! Statistics helpers: streaming moments, count distributions, container algorithms, compact count
//! encodings and memoisation.
//!
//! Ported from C++ coretools (`Math/TMeanVar*.h`, `Counters/*`, `algorithms.h`, `Types/TLogInt.h`,
//! `Types/TPseudoInt.h`, `memoize.h`):
//!
//! | C++ | here |
//! |---|---|
//! | `TMeanVar`, `TMeanVarMap`, `TMeanVarVector` | [`MeanVar`], [`MeanVarMap`], [`MeanVarVec`] |
//! | `TCountDistribution`, `TCountDistributionVector`, `TCountDistributionMap`, `TCountDistributionMultiDimensional` | [`CountDistribution`], [`CountDistributionVec`], [`CountDistributionMap`], [`CountDistributionArray`] |
//! | `ranks`, `pearsonCorrelation`, `spearmanCorrelation` | [`ranks`], [`pearson_correlation`], [`spearman_correlation`] |
//! | `normalize`, `normalizeLogSumExp` | [`normalize`], [`log_sum_exp`], [`normalize_log_sum_exp`] |
//! | `medianOfUnsorted`, `percentileOfUnsorted` | [`median`], [`percentile`] |
//! | `calculateMovingAverage`, `calculateMovingMedian` | [`moving_average`], [`moving_median`] |
//! | `getLinearIndex`, `getLinearIndexColMajor` | [`linear_index`], [`linear_index_col_major`] |
//! | `getSubscripts`, `getSubscriptsAsArray`, `getSubscriptsColMajor` | [`subscripts`], [`subscripts_array`], [`subscripts_col_major`] |
//! | `bisect` | [`bisect`] |
//! | `TLogInt`, `TPseudoInt` | [`LogInt`], [`PseudoInt`] |
//! | `TMemoizer`, `memoize` | [`Memoizer`] |
//!
//! The rest of `algorithms.h` is covered by std, statrs and ndarray: `mean`, `meanVar`, `var`, `sd`
//! (population moments) by [`MeanVar`] or statrs `Statistics::population_variance` /
//! `population_std_dev`; `containerSum`, `containerProduct`, `numNonZero`, `weightedSum`,
//! `sumPairwiseProduct`, `sumOfSquares`, `vectorNorm`, `euclideanDistance`, `pairwiseSum`, `reset`,
//! `normalize` by a given total, `standardizeZeroMean*`, `fillCumulative`, `fillFromNormalized`,
//! `findDuplicate` and the variadic `min` / `max` by iterator adapters; `rankSort` and
//! `sortContainerByRank` by `sort_by` on indices; `binarySearch_getIndex` by `binary_search` /
//! `partition_point`; `delta` by `abs_diff`; `getIncrements*` by ndarray strides. The rounding
//! variants of the moving average and median are `f64::round` on their results.
//! `CountDistribution*::write*` with file names or headers are the methods writing to any
//! [`std::io::Write`] after a header line. `TCountDistributionMultiDimensional::clear` and `resize`
//! are a new [`CountDistributionArray`].

mod algorithms;
mod compact;
mod count_distribution;
mod layout;
mod mean_var;
mod memoize;

pub use algorithms::{
    bisect, log_sum_exp, median, moving_average, moving_median, normalize, normalize_log_sum_exp,
    pearson_correlation, percentile, ranks, spearman_correlation,
};
pub use compact::{LogInt, PseudoInt};
pub use count_distribution::{
    CountDistribution, CountDistributionArray, CountDistributionMap, CountDistributionVec, Zeros,
};
pub use layout::{
    linear_index, linear_index_col_major, subscripts, subscripts_array, subscripts_col_major,
};
pub use mean_var::{MeanVar, MeanVarMap, MeanVarVec};
pub use memoize::Memoizer;
