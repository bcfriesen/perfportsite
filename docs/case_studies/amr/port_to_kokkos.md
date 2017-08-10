# Porting BoxLib to Kokkos

The fundamental data container in BoxLib is the FAB ("Fortran array box"). A
FAB contains the field data within a box on the problem domain. Each FAB
correlates to a unique box on the domain. The `Box` object in BoxLib is
primarily a metadata container, tracking geometric information about the box,
as well as its position on the domain, its intersection with other boxes, ghost
zone and boundary condition information, coarse-fine interfaces on refined
grids, etc.

A FAB is an array of data (typically of type `double` or `int` in C++) which is
usually accessed only in Fortran kernels. It is designed with Fortran data
access patterns (i.e., column-major) in mind. The BoxLib C++ framework itself
rarely manipulates raw FAB data. A large amount of infrastructure in BoxLib is
built around the `Box` and `FAB` objects. Consequently, achieving high
performance with a Kokkos implementation requires that the `Box` and `FAB`
objects be ported (or at least become compatible) with the Kokkos containers,
particularly the `View` object.

## Rewriting Fortran kernels in C++

Early on in the porting process we encountered significant challenges. The
first was that Kokkos has no support for Fortran. This meant that all of the
Fortran kernels in the geometric multigrid solver in BoxLib (which number
around 10, each containing 10s to 100s of lines of code) needed to be rewritten
in C or C++ before the porting process could begin at all. This was a
challenging and time-consuming undertaking, in part because the 3-D indexing
scheme into the FAB object, which is quite natural in Fortran (see the example
in the [code layout](./code_layout.md) page), becomes convoluted and
error-prone in C++. For example, the following Fortran subroutine, which
performs the restriction operation described [here](./multigrid.md):

```Fortran
 subroutine FORT_AVERAGE (
$     c, DIMS(c),
$     f, DIMS(f),
$     lo, hi, nc)
 implicit none
 integer nc
 integer DIMDEC(c)
 integer DIMDEC(f)
 integer lo(BL_SPACEDIM)
 integer hi(BL_SPACEDIM)
 REAL_T f(DIMV(f),nc)
 REAL_T c(DIMV(c),nc)

 integer i, i2, i2p1, j, j2, j2p1, k, k2, k2p1, n

 do n = 1, nc
    do k = lo(3), hi(3)
       k2 = 2*k
       k2p1 = k2 + 1
   do j = lo(2), hi(2)
          j2 = 2*j
          j2p1 = j2 + 1
          do i = lo(1), hi(1)
             i2 = 2*i
             i2p1 = i2 + 1
             c(i,j,k,n) =  (
$                 + f(i2p1,j2p1,k2  ,n) + f(i2,j2p1,k2  ,n)
$                 + f(i2p1,j2  ,k2  ,n) + f(i2,j2  ,k2  ,n)
$                 + f(i2p1,j2p1,k2p1,n) + f(i2,j2p1,k2p1,n)
$                 + f(i2p1,j2  ,k2p1,n) + f(i2,j2  ,k2p1,n)
$                 )*eighth
          end do
       end do
    end do
 end do

 end
```

becomes the following in C++:

```C++
void C_AVERAGE(const Box* bx,
        const int ng,
        const int nc,
        Real* c,
        const Real* f){


    int i2, j2, k2;
    int ijkn;
    int i2p1_j2p1_k2_n, i2_j2p1_k2_n, i2p1_j2_k2_n, i2_j2_k2_n;
    int i2p1_j2p1_k2p1_n, i2_j2p1_k2p1_n, i2p1_j2_k2p1_n, i2_j2_k2p1_n;

    const int ng;
    const int BL_jStride = bx->length(0) + 2*ng;
    const int BL_j2Stride = 2*(bx->length(0)) + 2*ng;
    const int BL_kStride = BL_jStride * (bx->length(1) + 2*ng);
    const int BL_k2Stride = BL_j2Stride * (2*(bx->length(1)) + 2*ng);
    const int BL_nStride = BL_kStride * (bx->length(2) + 2*ng);
    const int BL_n2Stride = BL_k2Stride * (2*(bx->length(2)) + 2*ng);
    const int *lo = bx->loVect();
    const int *hi = bx->hiVect();

    int abs_i, abs_j, abs_k;

      for (int n = 0; n<nc; n++){
         for (int k = 0; k < bx->length(2); ++k) {
            k2 = 2*k;
            for (int j = 0; j < bx->length(1); ++j) {
               j2 = 2*j;
               for (int i = 0; i < bx->length(0); ++i) {
                  i2 = 2*i;

                  abs_i = lo[0] + i;
                  abs_j = lo[1] + j;
                  abs_k = lo[2] + k;

                  ijkn = (i + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride + n*BL_nStride;

                  i2_j2_k2_n =       (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                  i2p1_j2p1_k2_n =   (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                  i2_j2p1_k2_n =     (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                  i2p1_j2_k2_n =     (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                  i2p1_j2p1_k2p1_n = (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                  i2_j2p1_k2p1_n =   (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                  i2p1_j2_k2p1_n =   (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                  i2_j2_k2p1_n =     (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;

                  c[ijkn] =  (f[i2p1_j2p1_k2_n] + f[i2_j2p1_k2_n] + f[i2p1_j2_k2_n] + f[i2_j2_k2_n])*(0.125);
                  c[ijkn] += (f[i2p1_j2p1_k2p1_n] + f[i2_j2p1_k2p1_n] + f[i2p1_j2_k2p1_n] + f[i2_j2_k2p1_n])*(0.125);
               }
            }
         }
      }
}
```

## Implementing execution spaces

After we ported all of the multigrid kernels to C++, we could then start adding
Kokkos to BoxLib. The simplest modification was to add Kokkos execution spaces
to each kernel. We accomplished this by converting each kernel from a regular C++ function to a functor, which Kokkos uses in its execution spaces.
The restriction kernel above became the following:

```C++
struct C_AVERAGE_Functor {
    // Data used by the loop body
    FArrayBox* c;
    FArrayBox* f;

    // Constructor to initialize the data
    C_AVERAGE_Functor(FArrayBox* c_, const FArrayBox* f_){
        c=c_;
        f=const_cast<FArrayBox*>(f_);
    }

    // Loop body as an operator
    KOKKOS_INLINE_FUNCTION
    void operator() (const int& n, const int& k, const int& j, const int& i) const {

      (*c)(IntVect(i,j,k),n) =  ((*f)(IntVect(2*i+1,2*j+1,2*k),n) +
                                 (*f)(IntVect(2*i,2*j+1,2*k),n) +
                                 (*f)(IntVect(2*i+1,2*j,2*k),n) +
                                 (*f)(IntVect(2*i,2*j,2*k),n))*0.125;

      (*c)(IntVect(i,j,k),n) += ((*f)(IntVect(2*i+1,2*j+1,2*k+1),n) +
                                 (*f)(IntVect(2*i,2*j+1,2*k+1),n) +
                                 (*f)(IntVect(2*i+1,2*j,2*k+1),n) +
                                 (*f)(IntVect(2*i,2*j,2*k+1),n))*0.125;
    }
};

void C_AVERAGE(
  const Box& bx,
  const int nc,
  FArrayBox& c,
  const FArrayBox& f){

    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();

    typedef Kokkos::Experimental::MDRangePolicy
      <Kokkos::Experimental::Rank
        <4,Kokkos::Experimental::Iterate::Right,Kokkos::Experimental::Iterate::Right>
      >
      t_policy;

    // Create a functor
    C_AVERAGE_Functor ave_functor(&c,&f);
    // Execute functor
    Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1],lo[0]},
                                                   {nc,hi[2]+1,hi[1]+1,hi[0]+1},
                                                   {1,4,4,1024000}),
                                          ave_functor);
}
```

Implementing functors in this way enabled these kernels to execute on CPUs
without requiring the `View` container. However, the performance was
substantially lower than the native C++ or Fortran kernels, and furthermore,
these kernels could not execute on accelerators because Kokkos can migrate data
between host and accelerator only with a `View`. Our next task, then was to
implement `View`s in BoxLib.


## Implementing `View`s

Changing the fundamental data container in BoxLib from FABs to Kokkos `View`s
was a formidable undertaking, and we enjoyed only partial success, even after
considerable effort.
##(GET DETAILS FROM ZAHRA AND THORSTEN)
