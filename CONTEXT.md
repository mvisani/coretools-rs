# coretools

Shared numerical and statistical building blocks (probabilities, distributions, positional data, multi-dimensional data files) for Rust programs that analyse genomic and other sequence-derived data. Rust successor of the C++ coretools library.

## Language

### Probabilities

**Probability representation**:
One of the encodings a probability can be stored in: Linear, Log, Log10, Phred or HpPhred.
_Avoid_: kind, type, scale

**Linear**:
A probability stored as-is, in `[0, 1]`.

**Log** / **Log10**:
A probability stored as its natural / base-10 logarithm, in `(-inf, 0]`.

**Phred**:
A probability stored as the integer `round(-10 * log10(p))`, in `0..=255`.
_Avoid_: quality score, Q-score

**HpPhred**:
A high-precision Phred: the integer `round(-1000 * log10(p))`, in `0..=65534`.

**Tag**:
A sentinel value stored in a probability slot in place of a probability; some representations can encode a byte in it.

### Values

**Bounded value**:
A number guaranteed to lie within a fixed interval (e.g. positive, strictly positive, open unit interval).
_Avoid_: weak type, strong type

### Positions

**Chunk**:
A named group of positions, such as a chromosome or contig.

**Distance group**:
The log2 bin of the distance between a position and the previous position in the same chunk; group 0 marks the first position of a chunk.

### Applications

**Task**:
A named unit of work an application exposes to its command line.
