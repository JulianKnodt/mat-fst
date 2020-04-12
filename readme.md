# Fixed Size Finite State Transducers Matrices

Finite State Transducers are a type of map, from a sequence of input elements to a value from
some output set.

If you squint your eyes, matrices are also a map from a series of indeces to some output value.
The indeces are always some fixed size, and all output types are uniform.

Usually, for dense matrices, coordinates are implicit, and be computed based on an elements
position in some 1D array which has certain
[strides](https://en.wikipedia.org/wiki/Stride_of_an_array) for each dimension. In the case of a
sparse matrix though, it is often the case that most of the output values are zero, and we waste
a lot of space representing them in the matrix. Instead of representing indeces implicitly, we
might choose to store values directly based on their index. This saves space, but often at the
cost of ease of computation, iteration, or the like. There exist a plethora of
[formats](https://en.wikipedia.org/wiki/Sparse_matrix) to represent sparse matrices which
attempt to optimize them for one purpose or another.

This is another attempt to create such a format.

This format uses the [FST](https://docs.rs/fst/0.4.0/fst/) library created by BurntSushi and
modifies multiple parts of it in attempt to provide a working version of a sparse matrix that
supports sparse vector multiplication. It also seeks to optimize other metrics as well, such as
allowing for efficient random-indexing, and a high-compression level, which are purely a result
of using the FST structure.

## Modifications from the existing FST library

TODO
- Fixed Size elements only
- Abstract over input and output types
- Iterators for values instead of the built in streams.
- Removing excess items
