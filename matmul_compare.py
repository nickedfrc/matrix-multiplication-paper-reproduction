# SPDX-License-Identifier: Apache-2.0
# Developed with OpenAI Codex assistance; see NOTICE for mathematical sources.
"""Dense floating-point matrix multiplication and reproducible CPU benchmarks.
Paper 9/4 bound: no explicit rank certificate is supplied, so not benchmarkable.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
import contextlib
import io
import json
from pathlib import Path
import platform
import random
import statistics
import time
import numpy as np

def inputs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected 2D inputs with matching inner dimensions")
    if a.dtype.kind not in "biufc" or b.dtype.kind not in "biufc":
        raise TypeError("Expected numeric inputs")
    dtype = np.result_type(a.dtype, b.dtype, np.float64)
    return a.astype(dtype, copy=False), b.astype(dtype, copy=False)

def numpy_matmul(a, b):
    a, b = inputs(a, b)
    return a @ b

def naive_matmul(a, b):
    a, b = inputs(a, b)
    m, k = a.shape
    n = b.shape[1]
    aa, bb = a.tolist(), b.tolist()
    c = [[0] * n for _ in range(m)]
    for i in range(m):
        for t in range(k):
            value = aa[i][t]
            for j in range(n):
                c[i][j] += value * bb[t][j]
    return np.asarray(c, dtype=a.dtype).reshape(m, n)

def outer_matmul(a, b):
    a, b = inputs(a, b)
    c = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
    for t in range(a.shape[1]):
        c += a[:, t:t+1] * b[t:t+1, :]
    return c

def blocked_matmul(a, b, block_size=128):
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    a, b = inputs(a, b)
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=a.dtype)
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for t in range(0, k, block_size):
                c[i:i+block_size, j:j+block_size] += (
                    a[i:i+block_size, t:t+block_size]
                    @ b[t:t+block_size, j:j+block_size])
    return c

def _strassen(a, b, leaf):
    n = a.shape[0]
    if n <= leaf:
        return a @ b
    h = n // 2
    a11,a12,a21,a22 = a[:h,:h],a[:h,h:],a[h:,:h],a[h:,h:]
    b11,b12,b21,b22 = b[:h,:h],b[:h,h:],b[h:,:h],b[h:,h:]
    p1 = _strassen(a11+a22, b11+b22, leaf)
    p2 = _strassen(a21+a22, b11, leaf)
    p3 = _strassen(a11, b12-b22, leaf)
    p4 = _strassen(a22, b21-b11, leaf)
    p5 = _strassen(a11+a12, b22, leaf)
    p6 = _strassen(a21-a11, b11+b12, leaf)
    p7 = _strassen(a12-a22, b21+b22, leaf)
    c = np.empty_like(a)
    c[:h,:h] = p1+p4-p5+p7
    c[:h,h:] = p3+p5
    c[h:,:h] = p2+p4
    c[h:,h:] = p1-p2+p3+p6
    return c

def strassen_matmul(a, b, leaf_size=128):
    """O(N**log2(7)); rectangular inputs padded to next-power-of-two square."""
    if not isinstance(leaf_size, int) or leaf_size < 1:
        raise ValueError("leaf_size must be a positive integer")
    a, b = inputs(a, b)
    m,k = a.shape
    n = b.shape[1]
    if min(m,k,n) == 0:
        return np.zeros((m,n), dtype=a.dtype)
    if m == k == n and (n & (n-1)) == 0:
        return _strassen(a,b,leaf_size)
    size = 1 << (max(m,k,n)-1).bit_length()
    ap = np.zeros((size,size), dtype=a.dtype)
    bp = np.zeros_like(ap)
    ap[:m,:k],bp[:k,:n] = a,b
    return _strassen(ap,bp,leaf_size)[:m,:n].copy()

def verify():
    rng = np.random.default_rng(20261009)
    methods = {"numpy":numpy_matmul, "naive":naive_matmul, "outer":outer_matmul,
               "blocked_3":lambda a,b:blocked_matmul(a,b,3),
               "strassen_1":lambda a,b:strassen_matmul(a,b,1),
               "strassen_4":lambda a,b:strassen_matmul(a,b,4)}
    cases,worst = 0,{name:0.0 for name in methods}
    shapes = [(0,3,2),(3,0,2),(3,2,0),(1,1,1),(2,2,2),
              (3,3,3),(7,5,9),(16,16,16),(31,17,23),(33,33,33)]
    for dtype in (np.float32,np.float64,np.complex128,np.int64):
        for m,k,n in shapes:
            a = rng.integers(-4,5,(m,k)).astype(dtype)
            b = rng.integers(-4,5,(k,n)).astype(dtype)
            if np.issubdtype(dtype,np.complexfloating):
                a += 1j*rng.normal(size=a.shape)
                b += 1j*rng.normal(size=b.shape)
            elif np.issubdtype(dtype,np.floating):
                a += rng.normal(size=a.shape)
                b += rng.normal(size=b.shape)
            reference = a.astype(np.result_type(dtype,np.float64)) @ b
            for name,fn in methods.items():
                result = fn(a,b)
                np.testing.assert_allclose(result,reference,rtol=1e-10,atol=1e-10)
                error = float(np.linalg.norm(result-reference)/max(float(np.linalg.norm(reference)),1.0))
                worst[name] = max(worst[name],error)
                cases += 1
    for i in range(4):
        for j in range(4):
            a,b = np.zeros((2,2)),np.zeros((2,2))
            a.flat[i],b.flat[j] = 1,1
            np.testing.assert_array_equal(strassen_matmul(a,b,1),a@b)
            cases += 1
    a,b = rng.normal(size=(18,18))[::2,::2],rng.normal(size=(9,9)).T
    for scale in (1e-50,1.0,1e50):
        for fn in methods.values():
            np.testing.assert_allclose(fn(a*scale,b/scale),a@b,rtol=1e-10,atol=1e-10)
            cases += 1
    for fn in methods.values():
        for a,b in [(np.ones(3),np.ones((3,2))),(np.ones((2,3)),np.ones((2,2)))]:
            try:
                fn(a,b)
            except ValueError:
                cases += 1
            else:
                raise AssertionError("Invalid shape accepted")
    return {"status":"passed","checks":cases,"worst_relative_errors":worst}

def timed(fn,a,b,repeats,target):
    result = fn(a,b)
    start = time.perf_counter()
    fn(a,b)
    estimate = max(time.perf_counter()-start,1e-9)
    batch = max(1,min(5000,int(target/estimate)))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(batch):
            result = fn(a,b)
        samples.append((time.perf_counter()-start)/batch)
    return result,samples,batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",type=int,nargs="+",default=[32,64,128,256,512,1024])
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--target-seconds",type=float,default=0.025)
    parser.add_argument("--naive-max",type=int,default=128)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    if min(args.sizes)<1 or args.repeats<1 or args.target_seconds<=0:
        parser.error("sizes, repeats and target-seconds must be positive")
    checks = verify()
    print(json.dumps(checks,indent=2),flush=True)
    if args.verify_only:
        return
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        np.show_config()
        np.show_runtime()
    data = {"paper_algorithm_status":"No explicit u, r, or coefficient rank decomposition in paper; not benchmarked",
            "environment":{"python":platform.python_version(),"numpy":np.__version__,
                           "platform":platform.platform(),"cpu":platform.processor(),
                           "thread_env":{k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                           "numpy_runtime":runtime.getvalue()},
            "config":{"seed":20261009,"dtype":"float64","sizes":args.sizes,
                      "repeats":args.repeats,"target_seconds":args.target_seconds,
                      "includes":"validation, conversion, allocation and padding; excludes generation and accuracy checks"},
            "verification":checks,"rows":[]}
    rng,order_rng = np.random.default_rng(20261009),random.Random(20261009)
    for n in args.sizes:
        a,b = rng.normal(size=(n,n)),rng.normal(size=(n,n))
        reference = a@b
        methods = {"numpy":numpy_matmul,"outer":outer_matmul,"blocked_128":blocked_matmul}
        if n <= args.naive_max:
            methods["naive"] = naive_matmul
        for leaf in (32,64,128,256):
            if leaf < n:
                methods[f"strassen_{leaf}"] = lambda a,b,leaf=leaf:strassen_matmul(a,b,leaf)
        order = list(methods)
        order_rng.shuffle(order)
        for name in order:
            result,samples,batch = timed(methods[name],a,b,args.repeats,args.target_seconds)
            np.testing.assert_allclose(result,reference,rtol=1e-9,atol=1e-9)
            error = float(np.linalg.norm(result-reference)/np.linalg.norm(reference))
            row = {"n":n,"method":name,"median_seconds":statistics.median(samples),
                   "min_seconds":min(samples),"max_seconds":max(samples),"samples_seconds":samples,
                   "calls_per_sample":batch,"relative_frobenius_error_vs_numpy":error}
            data["rows"].append(row)
            print(f"{n:4d} {name:14s} {row['median_seconds']*1000:10.4f} ms error={error:.2e}",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"benchmark.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    names = ["numpy","naive","outer","blocked_128","strassen_32","strassen_64","strassen_128","strassen_256"]
    lines = ["# Measured matrix multiplication benchmark","",
             "Median milliseconds; blank = omitted. Paper 9/4 algorithm has no runnable certificate.","",
             "| n | "+" | ".join(names)+" |","|---|"+"---:|"*len(names)]
    for n in args.sizes:
        table = {r["method"]:r for r in data["rows"] if r["n"]==n}
        values = [f"{table[name]['median_seconds']*1000:.4f}" if name in table else "" for name in names]
        lines.append(f"| {n} | "+" | ".join(values)+" |")
    lines += ["",f"Verification: {checks['checks']} checks passed.","",
              "Strassen leaves use BLAS; only leaf sizes smaller than n are timed.",
              "See JSON for raw samples, errors, environment and backend details."]
    (args.output/"benchmark.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__ == "__main__":
    main()

