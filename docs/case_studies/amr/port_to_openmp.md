# Porting BoxLib to OpenMP 4.x

Since version 4.0, OpenMP has supported accelerator devices through data
offloading and kernel execution semantics. At face value it appears to be a
highly desirable approach for performance portability, as it requires fairly
non-invasive code modifications (chiefly through directives which are ignored
as comments if OpenMP is not activated during compilation).

BoxLib already contains a large amount of OpenMP in the C++ framework to
implement thread parallelization and loop tiling (see [here](./parallelism.md)
and [here](./code_layout.md) for more details). However, these directives are
limited to version 3.0 and older, and consist primarily of simple
multi-threading of loops, such that the Fortran kernel execution happens
entirely within a thread-private region. This approach yields high performance
on self-hosted systems such as Intel Xeon and Xeon Phi, but provides no support
for architectures featuring a discrete accelerator such as a GPU.

We implemented the OpenMP `target` construct in several of the geometric
multigrid kernels in BoxLib in order to support kernel execution on GPUs.
