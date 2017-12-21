# RAJA Implementation

Similarly to Kokkos, RAJA abstracts loop parallelism into "execution models".
Unlike Kokkos, however, RAJA implements execution models and memory models (how
and where the data is placed in memory) separately; for this reason, porting a
code to RAJA can be done incrementally, resulting in potentially a more
palatable task than implementing Kokkos, which requires significant and
simultaneous changes throughout the code.

In the RAJA execution model, one can define several encapsulation features for
a loop:

  * _execution policy_: how the loop is executed (e.g., in which order, using
    which type of parallism)
  * _IndexSet_: the iteration space of the loop; a compiler can use this
    information to optimize the execution of the loop, e.g., by recognizing
    that each iteration has a data dependency only 1 iteration away, but not 2
    or more iterations away
  * _data type encapsulation_: contains alignment and aliasing attributes, as
    well as reduction types if needed

More information about RAJA is provided in the [documentation](https://software.llnl.gov/RAJA/).

Because RAJA implements the execution model and the memory model separately,
one can port the loops in a C or C++ code to the RAJA execution model without
changing how memory is allocated or arranged. This enables incremental
implementation of RAJA into existing codes. (However, it also means that the
loops cannot be executed on a discrete accelerator; to enable that one must
implement the [CHAI](https://github.com/LLNL/CHAI) memory model.)

## Porting the execution model

In BoxLib, porting the C++ multigrid kernels to the RAJA execution model is
straightforward. For example, the `C_AVERAGE()` kernel, which restricts a fine
grid onto a coarse grid, looks like the following in its native C++ form:

```C++
void C_AVERAGE(const Box& bx,
               const int nc,
               FArrayBox& c,
               const FArrayBox& f){

  const int *lo = bx.loVect();
  const int *hi = bx.hiVect();

  for (int n = 0; n<nc; n++){
    for (int k = lo[2]; k <= hi[2]; ++k) {
      int k2 = 2*k;
      for (int j = lo[1]; j <= hi[1]; ++j) {
        int j2 = 2*j;
        for (int i = lo[0]; i <= hi[0]; ++i) {
          int i2 = 2*i;

          c(IntVect(i,j,k),n) =  (f(IntVect(i2+1,j2+1,k2),n) + f(IntVect(i2,j2+1,k2),n) + f(IntVect(i2+1,j2,k2),n) + f(IntVect(i2,j2,k2),n))*0.125;
          c(IntVect(i,j,k),n) += (f(IntVect(i2+1,j2+1,k2+1),n) + f(IntVect(i2,j2+1,k2+1),n) + f(IntVect(i2+1,j2,k2+1),n) + f(IntVect(i2,j2,k2+1),n))*0.125;
        }
      }
    }
  }
}
```
When converted to RAJA, it is the following:
```C++
void C_AVERAGE(const Box& bx,
const int nc,
FArrayBox& c,
const FArrayBox& f){

        const int *lo = bx.loVect();
        const int *hi = bx.hiVect();

        RAJA::RangeSegment iBounds(lo[0], hi[0]+1);
        RAJA::RangeSegment jBounds(lo[1], hi[1]+1);
        RAJA::RangeSegment kBounds(lo[2], hi[2]+1);
        RAJA::RangeSegment nBounds(0, nc);

        // Since we are modifying the FAB "c" inside the loop, we need to do
        // the lambda capture by reference using [&] rather than [=].
        RAJA::forallN<RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>> (
            iBounds, jBounds, kBounds, nBounds, [&](int i, int j, int k, int n) {

              int i2 = 2*i;
              int j2 = 2*j;
              int k2 = 2*k;

              c(IntVect(i,j,k),n) =  (f(IntVect(i2+1,j2+1,k2),n) + f(IntVect(i2,j2+1,k2),n) + f(IntVect(i2+1,j2,k2),n) + f(IntVect(i2,j2,k2),n))*0.125;
              c(IntVect(i,j,k),n) += (f(IntVect(i2+1,j2+1,k2+1),n) + f(IntVect(i2,j2+1,k2+1),n) + f(IntVect(i2+1,j2,k2+1),n) + f(IntVect(i2,j2,k2+1),n))*0.125;

        });
}
```
The `RangeSegment` type is a special kind of IndexSet which contains all
possible values between the first and second arguments of the constructor.
RangeSegments are a common type of IndexSet in BoxLib calculations, which often
require looping over all possible grid points in order to perform an operation.

Because this is a 3D structured grid algorithm, we use the `forallN` and
`NestedPolicy` constructs to generate the execution policy for nested loop
iterations. The above snippet, we have requested sequential execution
(`seq_exec`) at all levels of the loop nest. One can combine the execution
policies available in RAJA arbitrarily among loop nest levels. On architectures
such as Xeon Phi, with wide vector units, one could instead use something like

```C++
RAJA::forallN<RAJA::NestedPolicy<
  RAJA::ExecList<RAJA::simd_exec, RAJA::seq_exec, RAJA::seq_exec, RAJA::omp_parallel_for_exec>>> (
    iBounds, jBounds, kBounds, nBounds, [&](int i, int j, int k, int n) {
```

which would use OpenMP parallelism on the outermost level of the loop nest (the
`n`-loop), sequential iteration on the 2 middle layers (the `k`- and
`j`-loops), and SIMD execution on the innermost loop (the `i`-loop). However,
in BoxLib this is undesirable because the multigrid kernels are called inside
thread-private regions; the above code would lead to nested threading, which
would likely degrade performance. In BoxLib, then, the optimal choice is typically:

```C++
RAJA::forallN<RAJA::NestedPolicy<
  RAJA::ExecList<RAJA::simd_exec, RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>> (
    iBounds, jBounds, kBounds, nBounds, [&](int i, int j, int k, int n) {
```

which applies SIMD parallelism to the innermost loop, and sequential execution
to all other levels of the loop nest.

Most of the other loops in the BoxLib multigrid algorithm can be ported in
similar ways. The only loop which is not straightforward to convert is the
Gauss-Seidel red-black (GSRB) smoothing kernel, which is the most
computationally expensive kernel in the algorithm. The innermost loop in GSRB
uses a stride-2 pattern which alternates between odd and even cells on each
pencil of constant (y,z), forming a 3-D "checkerboard" pattern. Because the
loop bounds of the x-loop depend on the value of y and z, as well as a
parameter which determines if one is iterating over "red" cells or "black"
cells, its `IndexSet` is not defined outside the (x,y,z) loop. Consequently,
one must divide the GSRB loop into two parts: an outer loop which iterates over
components and the (y,z) dimensions; and an inner loop which iterates over the
x dimension. The outermost loops appear as follows:

```C++
        RAJA::RangeSegment jBounds(lo[1], hi[1]+1);
        RAJA::RangeSegment kBounds(lo[2], hi[2]+1);
        RAJA::RangeSegment nBounds(0, nc);

        RAJA::forallN<RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>> (
            jBounds, kBounds, nBounds, [&](int j, int k, int n) {

              // ...

            });
```

Then, for a given value of `j` and `k`, one builds the index set for the inner
loop over the x-dimension, and then constructs a second RAJA loop:

```C++
        RAJA::forall<RAJA::seq_exec> (iBounds, [&](int i) {

          // apply operator stencil

        });
```

This approach is similar to the [Kokkos implementation](kokkos_implementation)
of the GSRB kernel.
