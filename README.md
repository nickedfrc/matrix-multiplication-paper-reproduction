# Matrix multiplication: finite constructions, FFT, and rank-certificate search

Independent Python reproduction of finite constructions from **An Upper Bound
of 9/4 for the Matrix Multiplication Exponent**, OpenAI, October 2, 2026.

[Paper and upstream source](https://github.com/openai/math/tree/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026)

Developed with **OpenAI Codex assistance**. Mathematical results belong to their
cited authors. This repository is an independent educational implementation.

## What is implemented

- Exact verification and recursive execution of algebraic rank certificates.
- A fair exhaustive search for the coefficients missing from the paper.
- FFT evaluation/interpolation for polynomial multiplication.
- FFT acceleration of the paper's finite Fourier projection.
- A working FFT implementation of matrix multiplication using polynomial packing.
- NumPy, Python loops, outer products, blocked multiplication, and Strassen baselines.

**No rank certificate achieving the 9/4 bound has been found in our experiments.
There is no measured 9/4 matrix multiplication result in this repository.**

The paper proves that, for every positive epsilon, matrix multiplication over
the complex numbers admits an arithmetic cost of O_epsilon(n^(9/4+epsilon)).
Its final recursion requires a fixed block size u and an exact rank-r
decomposition of the matrix multiplication tensor T_u. The paper does not list
a usable u, r, or coefficient arrays U, V, W.

## FFT: which coefficients does it compute?

For polynomial multiplication C(a,b), let L=a+b-1 and zeta=exp(-2*pi*i/L).
The evaluation/interpolation factors are explicit:

    U[s,i] = zeta^(s*i)
    V[s,j] = zeta^(s*j)
    W[k,s] = zeta^(-s*k) / L

These give the convolution coefficient identity:

    C[i,j,k] = sum_s W[k,s] U[s,i] V[s,j]

FFT applies these transformations efficiently without constructing the dense
factor matrices. The functions polynomial_rank_factors and polynomial_fft_mul
implement and test this identity.

In Proposition 3.1, the finite Fourier selector is:

    (1/L) sum_r exp(2*pi*i*r*q/L),  L=5M
    q = u-v+2(g-h)

It is computed using ifft(ones(L))[q mod L]. This replaces the direct sum over
Fourier copies in paper_constructions.py. The M^4 selector output still takes
O(M^4) space and work to materialize.

These convolution and projection factors are different from the unknown
rank decomposition of T_u needed for fast general matrix multiplication.
A fast FFT step does not automatically supply that decomposition.

## A working FFT matrix multiplication algorithm

fft_packed_matmul computes A of shape (m,k) times B of shape (k,n). It places:

    A[i,t] at polynomial exponent i*n*k+t
    B[t,j] at polynomial exponent j*k+k-1-t

The coefficient at (i*n+j)*k+k-1 is exactly the desired C[i,j]. The other
inner-index differences cannot collide with a requested coefficient because
their magnitude is smaller than k.

The implementation uses cyclic convolution with FFT length at least m*n*k.
The full product has degree at most m*n*k+k-2; any wrapped coefficients land
below k-1, while all requested coefficients start at k-1. Thus those aliases
do not affect the matrix product.

For square matrices this implementation uses O(n^3 log n) time and O(n^3)
storage. It computes real matrix products correctly, but does not achieve the
paper's 9/4 bound. It also supports complex inputs and irregular dimensions.

Measured on October 9, 2026, float64, three samples per method:

| Matrix size | NumPy, ms | FFT packing, ms | Strassen leaf 32, ms | FFT points |
|---|---:|---:|---:|---:|
| 32 | 0.0072 | 1.4901 | 0.0083 | 32,768 |
| 64 | 0.0191 | 20.9855 | 0.1039 | 262,144 |
| 128 | 0.1189 | 199.2055 | 0.7169 | 2,097,152 |

The Strassen row at size 32 is a leaf-only control; it performs no recursion.
At size 128, FFT packing is slower than NumPy despite its accurate result.
The default FFT resource limit is 2^21 points.

For the Proposition 3.1 selector at M=16, the direct root sum took 342.44 ms,
while FFT selection took 2.80 ms. This is a speedup of that finite projection,
not a speedup of complete matrix multiplication.

See results/fft_benchmark.md and results/fft_benchmark.json for the recorded
timings, raw samples, accuracy, and comparison of projection methods.

## Synthesizing the missing rank coefficients

synthesize_rank.py adds an exhaustive computability layer using Remark 5.2
of the paper. It is not a practical optimization method supplied by the paper.

All coefficients of one algebraic certificate lie in a common number field.
Represent them as rational polynomials in t modulo a monic rational
polynomial f(t). The code implements this arithmetic with Python Fraction;
no symbolic algebra package is required.

The search enumerates:

1. Block sizes u and admissible ranks r.
2. Monic rational polynomials f.
3. Rational coefficient vectors for U, V, W.

The schedule introduces one new task per round and advances every active task.
This avoids spending all search time on one block size that has no solution.
Every finite task and coefficient description is eventually visited in
unlimited mode. Identities are checked exactly modulo f, rather than accepted
because a floating residual happens to be small.

For a positive rational epsilon, the admission test is:

    r < u^(9/4 + epsilon/2)

It uses integer powers to avoid floating-point decisions. Conditional on the
paper's existence theorem and algebraic-coefficient observation, unlimited
search eventually finds a qualifying certificate. The proof provides no useful
bound on the required preprocessing time, memory, or block size. Exhaustive
enumeration is astronomically expensive and should not be treated as a
practical fast-matrix-multiplication implementation.

Reducible f is allowed: an identity modulo f remains true after evaluation at
any complex root of f. The enumeration includes minimal polynomials, so it also
covers the common-number-field certificates required by the argument.

Once a certificate is found, multiply_exact executes the recursive scheme on
rational matrix inputs using exact quotient-algebra arithmetic. numerical
provides a complex128 adapter; this adapter is approximate.

The synthesis tests recover a scalar rank-one decomposition through actual
enumeration, verify rational and algebraic versions of the known Strassen
certificate, reject a corrupted certificate, and test exact recursive products.
**Strassen is a test witness, not a discovered 9/4 certificate.**
A bounded unsuccessful search does not refute the paper's theorem.

## Files

| File | Purpose |
|---|---|
| matmul_compare.py | Classical baselines, Strassen, numerical checks, timing |
| fft_algorithms.py | Polynomial FFT factors, Fourier selector, packed FFT matrix multiplication |
| synthesize_rank.py | Exact algebraic certificates, fair search, exact recursion |
| bilinear_scheme.py | Generic complex128 executor for supplied U,V,W |
| paper_constructions.py | Finite tensor identities from Sections 3-5 |
| check_threads.py | Read-only OpenBLAS thread query |
| results/benchmark.* | Original baseline timings and raw data |
| results/fft_benchmark.* | FFT timings and raw data |
| results/paper_checks.json | Finite-construction checks |
| results/synthesis_checks.json | Exact synthesis and recursion tests |
| results/synthesis_search.json | Additional bounded search report |

## Run

Python 3.10 or newer and NumPy are required.

    python -m pip install -r requirements.txt
    python matmul_compare.py --verify-only
    python paper_constructions.py
    python bilinear_scheme.py
    python synthesize_rank.py --demo
    python fft_algorithms.py
    python matmul_compare.py

Bounded coefficient search:

    python synthesize_rank.py --epsilon 1/10 --max-rounds 400 --max-seconds 10

Unlimited search, explicitly opt-in and potentially impractical:

    python synthesize_rank.py --epsilon 1/10 --unbounded

If a certificate is found, it is saved as results/synthesized_certificate.json.
It can be loaded and executed as follows:

    from synthesize_rank import ExactCertificate
    certificate = ExactCertificate.load("results/synthesized_certificate.json")
    assert certificate.qualifies("1/10")
    exact_result = certificate.multiply_exact([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    numerical_result = certificate.numerical().multiply(a, b, leaf_size=128)

If the search exhausts its budget, the program reports failure to find a
certificate and does not silently substitute Strassen for the requested bound.
Budgets are checked between candidates; one exact candidate check is not
preempted in the middle.

## Validation and benchmark scope

The original baselines passed 286 checks covering basis pairs, multiple
Strassen recursion levels, irregular shapes, empty dimensions, complex inputs,
noncontiguous views, scaling, and invalid shapes. The FFT suite adds 66 checks.
The finite construction suite checks six separation instances, 25 determinant
filtrations, and 30 three-sector decompositions.

The original 1024-square measurements were: NumPy 44.20 ms, blocked
multiplication 75.66 ms, and the fastest tested Strassen setting 97.81 ms
(leaf size 256). These measurements are specific to this machine and
implementation; finite timings do not establish asymptotic exponents.

Recorded environment: Windows, Python 3.12.14, NumPy 2.3.5, OpenBLAS 0.3.30.
BLAS thread variables are set to 1 before NumPy import. A separate fresh-process
OpenBLAS query returned one thread; the original benchmark process recorded
the environment setting rather than directly querying its thread count.

Timing includes validation, conversion, allocation, and padding or FFT packing.
It excludes input generation and accuracy comparison. NumPy serves as the
floating-point reference, not an independent high-precision truth source.
Exact certificate checks and rational recursive products use Fraction.

Floating matrix inputs are promoted to at least float64, or complex128.
They are unsuitable for exact large-integer arithmetic. Strassen pads
rectangular inputs to a square power of two, which can be expensive for thin
matrices.

Finite identity tests and the search code do not independently verify the
paper's entire existence proof or measure its asserted asymptotic algorithm.

## AI assistance, attribution, and license

Implementation, documentation, and local verification were developed with
OpenAI Codex assistance. Test execution does not imply human peer review.
The mathematical results are attributed to their authors; this project makes
no new matrix multiplication exponent claim and is not an official OpenAI
implementation.

The code was independently written from mathematical definitions. No upstream
Python or Lean source, paper PDF, extracted paper text, paper figures, or Zhihu
answer is redistributed. Only source links and necessary mathematical
references are included.

Original code and documentation are licensed under Apache License 2.0; see
LICENSE. Third-party materials retain their own rights, and this repository
does not relicense them. NumPy is an external dependency and is not vendored.
NOTICE records the mathematical and dependency sources.

