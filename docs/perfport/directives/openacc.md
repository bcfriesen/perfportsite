# OpenACC

OpenACC is a set of standardized, high-level pragmas that enable C/C++ and Fortran programmers
to exploit parallel (co)processors, especially GPUs. OpenACC pragmas can be used to annotate
codes to enable data location, data transfer, and loop or code block parallelism.

Though OpenACC has much in common with OpenMP, the syntax of the directives is different.
More importantly, OpenACC can best be described as having
a *descriptive* model, in contrast to the more *prescriptive* model presented by OpenMP.
This difference in philosophy can most readily be seen by, e.g.,  comparing the ``acc loop`` directive
to the OpenMP implementation of the equivalent construct. In OpenMP, the programmer has responsibility
to specify how the parallelism in a loop is distributed (e.g., via ``distribute`` and ``schedule`` clauses).
In OpenACC, the runtime determines how to decompose the iterations across gangs or workers and vectors.
At an even higher level, an OpenACC programmer can use the ``acc kernels`` construct to allow the compiler complete freedom
to map the available parallelism in a code block to the available hardware.




## OpenACC at a glance

Some of the most important  data and control clauses for two of the most
used constructs in OpenACC programming - ``$acc parallel`` and ``$acc kernels`` - are
listed below. The data placement and movement clauses also appear in ``$acc data`` constructs.
``$acc loop`` provides control of parallelism similarly to ``$acc parallel`` but provides loop-level control.

Much more detail can be found at:

* [openacc.org](https://www.openacc.org/)

* [OpenACC Best Practices Guide](www.openacc.org/sites/default/files/inline.../OpenACC_Programming_Guide_0.pdf)

* [NVIDIA OpenACC resources](https://developer.nvidia.com/openacc)

* [OLCF Accelerator Programming Guide; Directive Programming](https://www.olcf.ornl.gov/support/system-user-guides/accelerated-computing-guide/#371)

* [OLCF Accelerator Programming Tutorials](https://www.olcf.ornl.gov/support/tutorials/) (includes examples of interoperability with CUDA and GPU libraries like CuFFT)


|construct             | important clauses  | description |
|:---|:---|---:|
|``$acc parallel``
|    |`num_gangs(expression)`| Controls how many parallel gangs are created
|    |`num_workers(expression)`| Controls how many workers are created in each gang
|    |`vector_length(list)`| Controls vector length of each worker
|    |`private(list)`| A copy of each variable in list is allocated to each gang
|    |`firstprivate(list)`| private variables initialized from host
|    |`reduction(operator:list)`| private variables combined across gangs
|``$acc kernels`` |  |  |
| | `copy(list)`| Allocates memory on GPU and copies data from host to GPU when entering region and copies data to the host when exiting region
| | `copyin(list)` | Allocates memory on GPU and copies data from host to GPU when entering region
| | `copyout(list)` |  Allocates memory on GPU and copies data to the host when exiting region
| | `create(list)` | Allocates memory on GPU but does not copy
| | `present(list)` | Data is already present on GPU from another containing data region


## Optimizing compute kernels for CPUs and GPUs

OpenACC provides descriptive directives which the programmer can use to
indicate to the compiler that a particular kernel is desired to achieve high
performance. The programmer may then instruct the compiler to optimize the
decorated OpenACC kernels for a particular computer architecture, including
both CPUs and GPUs.

For example, one may start with a simple Jacobi iterative solver, as provided
in the [OpenACC GitHub
page](https://raw.githubusercontent.com/OpenACCUserGroup/openacc-users-group/master/Contributed_Sample_Codes/Tutorial1/solver/jsolvef.F90).
One may then decorate one of the loops in this code with the `acc kernels`
directive, which is a very general suggestion to the compiler that the kernel
which follows should be optimized for a particular compute architecture.

```Fortran
!$acc kernels
do i = 1, nsize
  rsum = 0
  do j = 1, nsize
    if( i /= j ) rsum = rsum + A(j,i) * xold(j)
  enddo
  xnew(i) = (b(i) - rsum) / A(i,i)
enddo
!$acc end kernels
```

One may then compile this code for a multi-core CPU with the following compile
stanza:

```console
pgfortran -o jacobi_CPU.ex -fast -Minfo=all -tp=skylake jacobi.F90 &> compile_CPU.log
```

which will print diagonstic messages indicating the optimizations made for the
Intel 'Skylake' CPU architecture:

```console
init_simple_diag_dom:
     48, Loop not vectorized/parallelized: contains call
     56, Zero trip check eliminated
         Generated vector simd code for the loop
main:
    106, Memory zero idiom, loop replaced by call to __c_mzero8
    107, Memory zero idiom, loop replaced by call to __c_mzero8
    108, Loop not vectorized/parallelized: contains call
    123, Loop not vectorized/parallelized: potential early exits
         FMA (fused multiply-add) instruction(s) generated
    131, Loop not fused: different loop trip count
    133, Zero trip check eliminated
    143, Loop not fused: function call before adjacent loop
         Generated vector simd code for the loop containing reductions
    164, Loop not fused: function call before adjacent loop
    166, Zero trip check eliminated
         Generated vector simd code for the loop containing reductions
```

One can compile the same code with different flags, targeting NVIDIA V100
'Volta' GPUs:

```console
pgfortran -o jacobi_GPU.ex -fast -Minfo=all -ta=tesla:cc70 -Mcuda=cc70,cuda10.1,lineinfo jacobi.F90 &> compile_GPU.log
```

which will yield a different set of diagnostic messages:

```console
init_simple_diag_dom:
     48, Loop not vectorized/parallelized: contains call
     56, Zero trip check eliminated
         Generated vector simd code for the loop
main:
    106, Memory zero idiom, loop replaced by call to __c_mzero8
    107, Memory zero idiom, loop replaced by call to __c_mzero8
    108, Loop not vectorized/parallelized: contains call
    123, Loop not vectorized/parallelized: potential early exits
         FMA (fused multiply-add) instruction(s) generated
    130, Generating implicit copyout(xnew(1:nsize))
         Generating implicit copyin(b(1:nsize),a(1:nsize,1:nsize))
         Generating implicit copyin(xold(1:nsize))
    131, Loop carried dependence of xnew prevents parallelization
         Loop carried backward dependence of xnew prevents vectorization
         Complex loop carried dependence of xold prevents parallelization
         Generating Tesla code
        131, !$acc loop seq
        133, !$acc loop vector(128) ! threadidx%x
        134, Generating implicit reduction(+:rsum)
    133, Loop is parallelizable
    143, Loop not fused: function call before adjacent loop
         Generated vector simd code for the loop containing reductions
    164, Loop not fused: function call before adjacent loop
    166, Zero trip check eliminated
         Generated vector simd code for the loop containing reductions
```

By setting the target compute architecture in the compiler invocation, one can
compile the same code to be optimized for each; this is a very useful feature
of OpenACC.

## How to use OpenACC on ASCR facilities

### OLCF

####Using C/C++

PGI Compiler

```
$ module load cudatoolkit
$ cc -acc vecAdd.c -o vecAdd.out
```

Cray Compiler

```
$ module switch PrgEnv-pgi PrgEnv-cray
$ module load craype-accel-nvidia35
$ cc -h pragma=acc vecAdd.c -o vecAdd.out
```

####Using Fortran

PGI Compiler

```
$ module load cudatoolkit
$ ftn -acc vecAdd.f90 -o vecAdd.out
```

Cray Compiler

```
$ module switch PrgEnv-pgi PrgEnv-cray
$ module load craype-accel-nvidia35
$ ftn -h acc vecAdd.f90 -o vecAdd.out
```

### NERSC

The PGI compilers are provided on the [Cori GPU
nodes](https://docs-dev.nersc.gov/cgpu) at NERSC via the `pgi` modules.

## Benefits and Challenges

### Benefits

* Available for many different languages
* Interoperable with other approaches (e.g. CUDA or OpenMP)
* Allows performance optimization
* Controlled by well-defined standards bodies

### Challenges

* Relatively few compiler implementations at present (versus OpenMP)
* Evolving standards
* Descriptive approach sometimes impedes very high performance for a given kernel
