# Divergence log

Behavioural differences between coretools-rs and C++ [wegmannlab/coretools](https://bitbucket.org/wegmannlab/coretools) at commit `126744e`. Every entry is pinned by a Rust test. API reshaping without a behavioural effect (traits instead of implicit conversions, `Result` instead of exceptions for user errors) is not listed; see ADR-0001.

| C++ unit | old behaviour | new behaviour | reason | Rust test |
|---|---|---|---|---|
| `TConverter<phred>::from<hpPhred>` | Floors: `t/100` is integer division, so HpPhred 699 → Phred 6 and 150 → 1. | Rounds to nearest: 699 → 7, 150 → 2. | Every other conversion to Phred rounds; flooring biased HpPhred → Phred towards higher probabilities. | `probability::tests::hp_phred_to_phred_rounds_to_nearest` |
| `TConverter<phred>` / `TConverter<hpPhred>` from Linear, Log, Log10 | Negative or NaN Phred scores (Linear input above 1, NaN input) are cast to an unsigned integer: undefined behaviour. | Map to 0, the highest probability. | Remove undefined behaviour. | `probability::tests::conversions_to_phred_saturate` |
| `TSomeProbability(value)` | Range check only when compiled with `CHECK_INTERVALS` (off by default); otherwise out-of-range values are accepted silently. | `Probability::new` always checks and returns `ProbabilityError::OutOfRange`; `new_unchecked` skips the check. | Invalid input must be caught where it enters, independent of build flags. | `probability::tests::linear_probability`, `probability::tests::log_probability`, `probability::tests::log10_probability` |
| `TSomeProbability::getAsTag` | On a value that is not a tag, asserts only with `CHECK_INTERVALS`; otherwise returns an arbitrary byte. | `byte_tag` returns `None`. | A non-tag has no byte to decode. | `probability::tests::byte_tags` |
| `average(TConstView<Probability>)` | An empty view divides by zero and returns NaN. | `Probability::average` returns `None`. | NaN is not a probability. | `probability::tests::average_stays_in_the_same_representation` |
| `fromChar` (probability.h) | Takes a signed `char`; bytes 128–255 are negative and clamp to `!` (Phred 0). | `PhredProbability::from_ascii` takes a `u8`; bytes above `~` clamp to `~` (Phred 93). | Clamp by byte value, consistently at both ends of `33..=126`. | `probability::tests::phred_ascii_base_quality_clamps` |
