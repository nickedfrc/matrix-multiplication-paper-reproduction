# SPDX-License-Identifier: Apache-2.0
# Developed with OpenAI Codex assistance; see NOTICE.
"""FFT implementations of Fourier projection and packed matrix multiplication.

The packed algorithm computes true matrix products, but uses O(m*k*n) FFT
storage and O(m*k*n*log(m*k*n)) operations; it is NOT the 9/4 algorithm.
"""
# Import first to use the benchmark's thread configuration before NumPy.
from matmul_compare import inputs, numpy_matmul, strassen_matmul, timed
import json
from pathlib import Path
import statistics
import time
import numpy as np

def fourier_project_fft(M):
    """Prop. 3.1 selector. ifft(ones)[q] = (1/L) sum_r exp(2*pi*i*r*q/L)."""
    if not isinstance(M,int) or M<1:
        raise ValueError("M must be a positive integer")
    L = 5*M
    labels = np.arange(1,M+1)
    h,g,u,v = np.meshgrid(labels,labels,labels,labels,indexing="ij")
    phase = u-v+2*(g-h)
    delta = np.fft.ifft(np.ones(L,dtype=complex))
    return delta[phase%L]

def fourier_project_direct(M):
    L = 5*M
    labels = np.arange(1,M+1)
    h,g,u,v = np.meshgrid(labels,labels,labels,labels,indexing="ij")
    phase = u-v+2*(g-h)
    return np.exp(2j*np.pi*np.arange(L)[:,None,None,None,None]*phase/L).mean(axis=0)

def fft_packed_matmul(a,b,max_fft_points=1<<21):
    """A_it at i*n*k+t; B_tj at j*k+k-1-t.
    Output C_ij is convolution coefficient (i*n+j)*k+k-1.

    Let P=m*n*k and L=next_power_of_two(P). The full product degree is
    at most P+k-2. With cyclic convolution, wrapped terms have indices at
    most k-2; desired indices start at k-1, so those aliases cannot affect
    any requested coefficient. Hence FFT length L>=P is sufficient.
    """
    a,b = inputs(a,b)
    m,k = a.shape
    n = b.shape[1]
    if min(m,k,n)==0:
        return np.zeros((m,n),dtype=a.dtype)
    points = 1 << (m*n*k-1).bit_length()
    if points>max_fft_points:
        raise ValueError(f"FFT requires {points} points, above configured limit {max_fft_points}")
    ap,bp = np.zeros(points,dtype=a.dtype),np.zeros(points,dtype=a.dtype)
    ia = np.arange(m)[:,None]*(n*k)+np.arange(k)[None,:]
    ib = np.arange(n)[None,:]*k+(k-1-np.arange(k)[:,None])
    ap[ia],bp[ib] = a,b
    if np.iscomplexobj(a):
        convolution = np.fft.ifft(np.fft.fft(ap)*np.fft.fft(bp))
    else:
        convolution = np.fft.irfft(np.fft.rfft(ap)*np.fft.rfft(bp),n=points)
    desired = (np.arange(m)[:,None]*n+np.arange(n)[None,:])*k+k-1
    return convolution[desired].copy()


def polynomial_rank_factors(a,b):
    """Explicit rank-(a+b-1) decomposition of polynomial convolution C(a,b).
    These are convolution coefficients, not matrix-multiplication certificates.
    """
    if not isinstance(a,int) or not isinstance(b,int) or min(a,b)<1:
        raise ValueError("Polynomial lengths must be positive integers")
    L = a+b-1
    s = np.arange(L)
    U = np.exp(-2j*np.pi*s[:,None]*np.arange(a)[None,:]/L)
    V = np.exp(-2j*np.pi*s[:,None]*np.arange(b)[None,:]/L)
    W = np.exp(2j*np.pi*np.arange(L)[:,None]*s[None,:]/L)/L
    return U,V,W

def polynomial_fft_mul(a,b):
    a,b = np.asarray(a),np.asarray(b)
    if a.ndim!=1 or b.ndim!=1 or not len(a) or not len(b):
        raise ValueError("Expected nonempty coefficient vectors")
    L = len(a)+len(b)-1
    result = np.fft.ifft(np.fft.fft(a,n=L)*np.fft.fft(b,n=L))
    return result if np.iscomplexobj(a) or np.iscomplexobj(b) else result.real

def verify():
    rng = np.random.default_rng(20261009)
    checks = 0
    for M in range(1,13):
        labels = np.arange(1,M+1)
        h,g,u,v = np.meshgrid(labels,labels,labels,labels,indexing="ij")
        expected = (u-v+2*(g-h)==0).astype(float)
        np.testing.assert_allclose(fourier_project_fft(M),expected,rtol=0,atol=1e-13)
        checks += 1
    for a_len in range(1,5):
        for b_len in range(1,5):
            U,V,W = polynomial_rank_factors(a_len,b_len)
            tensor = np.einsum("ks,si,sj->ijk",W,U,V)
            expected = np.zeros_like(tensor)
            for i in range(a_len):
                for j in range(b_len):
                    expected[i,j,i+j] = 1
            np.testing.assert_allclose(tensor,expected,rtol=0,atol=1e-12)
            aa,bb = rng.normal(size=a_len),rng.normal(size=b_len)
            np.testing.assert_allclose(polynomial_fft_mul(aa,bb),np.convolve(aa,bb),
                                       rtol=1e-10,atol=1e-10)
            checks += 2
    shapes = [(0,3,2),(3,0,2),(3,2,0),(1,1,1),(1,3,1),
              (2,2,2),(3,5,7),(9,7,5),(16,16,16),(31,17,23)]
    for complex_input in (False,True):
        for m,k,n in shapes:
            a,b = rng.normal(size=(m,k)),rng.normal(size=(k,n))
            if complex_input:
                a = a+1j*rng.normal(size=a.shape)
                b = b+1j*rng.normal(size=b.shape)
            np.testing.assert_allclose(fft_packed_matmul(a,b),a@b,rtol=1e-10,atol=1e-10)
            checks += 1
    a,b = rng.normal(size=(18,18))[::2,::2],rng.normal(size=(9,9)).T
    np.testing.assert_allclose(fft_packed_matmul(a,b),a@b,rtol=1e-10,atol=1e-10)
    checks += 1
    try:
        fft_packed_matmul(np.ones((8,8)),np.ones((8,8)),max_fft_points=32)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("FFT resource limit ignored")
    return checks

def main():
    checks = verify()
    rng = np.random.default_rng(20261009)
    rows,projectors = [],[]
    for n in (32,64,128):
        a,b = rng.normal(size=(n,n)),rng.normal(size=(n,n))
        reference = a@b
        methods = {"numpy":numpy_matmul,"fft_packed":fft_packed_matmul,
                   "strassen_leaf_32":lambda a,b:strassen_matmul(a,b,32)}
        for name,fn in methods.items():
            result,samples,batch = timed(fn,a,b,3,0.02)
            np.testing.assert_allclose(result,reference,rtol=1e-9,atol=1e-9)
            row = {"n":n,"method":name,"median_ms":statistics.median(samples)*1000,
                   "samples_seconds":samples,"calls_per_sample":batch,
                   "relative_error":float(np.linalg.norm(result-reference)/np.linalg.norm(reference))}
            if name=="fft_packed":
                row["fft_points"] = 1 << (n**3-1).bit_length()
            rows.append(row)
            print(json.dumps(row),flush=True)
    for M in (4,8,16):
        results = {}
        for name,fn in (("direct",fourier_project_direct),("fft",fourier_project_fft)):
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                result = fn(M)
                samples.append(time.perf_counter()-start)
            results[name] = {"median_ms":statistics.median(samples)*1000}
            if name=="direct":
                expected = result
            else:
                np.testing.assert_allclose(result,expected,rtol=0,atol=1e-12)
        projectors.append({"M":M,**results})
    report = {"status":"passed","checks":checks,"dtype":"float64","seed":20261009,
              "repeats":3,"thread_setting":"same pre-import settings as matmul_compare",
              "timing_includes":"validation, packing, FFT, allocation and extraction",
              "rows":rows,"fourier_projection":projectors,
              "complexity":"O(m*n*k*log(m*n*k)) time; O(m*n*k) storage",
              "fast_rank_certificate_found":False}
    output = Path(__file__).with_name("results")
    output.mkdir(exist_ok=True)
    (output/"fft_benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines = ["# FFT comparison","",
             "Measured median milliseconds. FFT packing is not the 9/4 algorithm.","",
             "| n | NumPy | FFT packing | Strassen leaf 32 | FFT points |",
             "|---|---:|---:|---:|---:|"]
    for n in (32,64,128):
        table = {r["method"]:r for r in rows if r["n"]==n}
        lines.append(f"| {n} | {table['numpy']['median_ms']:.4f} | {table['fft_packed']['median_ms']:.4f} | {table['strassen_leaf_32']['median_ms']:.4f} | {table['fft_packed']['fft_points']} |")
    lines += ["","## Proposition 3.1 Fourier selector","",
              "| M | Direct root sum | FFT selector |","|---|---:|---:|"]
    for row in projectors:
        lines.append(f"| {row['M']} | {row['direct']['median_ms']:.4f} | {row['fft']['median_ms']:.4f} |")
    lines += ["",f"{checks} functional checks passed; all timed outputs also checked.",
              "Projection acceleration does not synthesize a fast matrix rank decomposition.",
              "Strassen at n=32 with leaf=32 is a leaf-only control, not recursive Strassen."]
    (output/"fft_benchmark.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","checks":checks,"fourier_projection":projectors},indent=2))

if __name__=="__main__":
    main()

