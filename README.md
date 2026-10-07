# coretools-rs

Shared numerical and statistical building blocks (probabilities, distributions, positional data, multi-dimensional data files) for Rust programs that analyse genomic and other sequence-derived data.

## Origin

coretools-rs is a Rust port of the C++ library [wegmannlab/coretools](https://bitbucket.org/wegmannlab/coretools) by the Wegmann lab. The port follows upstream commit `126744e`; a reference checkout of that commit is kept in `external/coretools`, which is not tracked by this repository.

Behavioural differences from the C++ library are listed in [`docs/divergences.md`](docs/divergences.md).

## License

Licensed under the Mozilla Public License 2.0 (`MPL-2.0`), the license of the upstream C++ library; the full text is in the `LICENSE` file.
