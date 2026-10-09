# SPDX-License-Identifier: Apache-2.0
# Developed with OpenAI Codex assistance; see NOTICE for mathematical sources.
"""Executable finite constructions in paper.pdf, Sections 3--5.
These checks do not prove the complete theorem or implement n**(9/4+epsilon).
"""
import json
import math
from pathlib import Path
import numpy as np
from fft_algorithms import fourier_project_fft

def polynomial_tensor(a,b):
    c = np.zeros((a,b,a+b-1),dtype=np.int64)
    for i in range(a):
        for j in range(b):
            c[i,j,i+j] = 1
    return c

def separation(M, epsilon=0.0, coefficients=None):
    """Prop. 3.1. Compressed tensor axes: h,a,g,b,u,c,v.
    Y/Z sectors are matched already, so only one h axis is stored.
    The Fourier indicator is computed exactly after numerical cross-check.
    Do not evaluate separate negative powers of epsilon at epsilon=0.
    """
    if M<1 or not 0<=epsilon<=1:
        raise ValueError("M>=1 and epsilon in [0,1] required")
    if coefficients is None:
        coefficients = np.ones((M,1,1,1),dtype=np.float64)
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 4 or coefficients.shape[0]!=M:
        raise ValueError("coefficients must have axes h,a,b,c")
    labels = np.arange(1,M+1)
    h,g,u,v = np.meshgrid(labels,labels,labels,labels,indexing="ij")
    phase = u-v+2*(g-h)
    keep = phase==0
    weight = g*g+h*u-h*h-h*v
    assert np.all(weight[keep]==(g-h)[keep]**2)
    assert np.all(weight[keep]>=0)
    L = 5*M
    numerical = fourier_project_fft(M)
    np.testing.assert_allclose(numerical,keep,atol=2e-13,rtol=0)
    factors = np.zeros_like(phase,dtype=np.float64)
    factors[keep] = epsilon**weight[keep]
    # Retain a symbolic integer selector instead of cancellation-prone roots.
    tensor = np.einsum("habc,hguv->hagbucv",coefficients,factors)
    return tensor, float(np.max(np.abs(numerical-keep)))

def filtration_basis(e):
    """Columns: e+2 quotient lifts followed by e kernel vectors D*g."""
    B = np.zeros((2*(e+1),2*(e+1)),dtype=np.float64)
    for i in range(e+1):
        B[i,i] = 1
    B[2*e+1,e+1] = 1  # v**e*w
    for j in range(e):
        B[e+1+j,e+2+j] = 1  # u*w*g
        B[j+1,e+2+j] = -1   # -v*s*g
    return B

def determinant_filtration(a,b):
    """Lemma 4.1: simultaneously block-triangularize every X slice."""
    if a<1 or b<2:
        raise ValueError("a>=1,b>=2")
    e,f = b-1,a+b-2
    Bin,Bout = filtration_basis(e),filtration_basis(f)
    nq_in,nq_out = b+1,a+b
    plus,minus = polynomial_tensor(a,b+1),polynomial_tensor(a,b-1)
    worst,off_diagonal = 0.0,0
    for i in range(a):
        raw = np.zeros((2*(f+1),2*(e+1)))
        for leg in range(2):
            for j in range(e+1):
                raw[leg*(f+1)+i+j,leg*(e+1)+j] = 1
        converted = np.linalg.solve(Bout,raw@Bin)
        expected = np.zeros_like(converted)
        expected[:nq_out,:nq_in] = plus[i].T
        expected[nq_out:,nq_in:] = minus[i].T
        diagonal = converted.copy()
        off_diagonal += int(np.count_nonzero(diagonal[nq_out:,:nq_in]))
        diagonal[nq_out:,:nq_in] = 0
        np.testing.assert_allclose(diagonal,expected,atol=1e-12)
        worst = max(worst,float(np.max(np.abs(diagonal-expected))))
    return {"a":a,"b":b,"diagonal_error":worst,"discarded_positive_weight_entries":off_diagonal}

def three_sectors(a,h):
    """Lemma 4.2; exact integer support and weight-zero tensor."""
    B = 3*h+a-1
    c = polynomial_tensor(a,B)
    yw = np.zeros(B,dtype=np.int64)
    zw = np.zeros(a+B-1,dtype=np.int64)
    yw[h:2*h+a-1] = 1
    zw[h+a-1:2*h+a-1] = -1
    weights = yw[None,:,None]+zw[None,None,:]
    weights = np.broadcast_to(weights,c.shape)
    assert np.all(weights[c!=0]>=0)
    survived = np.where(weights==0,c,0)
    small = polynomial_tensor(a,h)
    left = survived[:,:h,:h+a-1]
    right = survived[:,2*h+a-1:,2*h+a-1:]
    middle = survived[:,h:2*h+a-1,h+a-1:2*h+a-1]
    np.testing.assert_array_equal(left,small)
    np.testing.assert_array_equal(right,small)
    np.testing.assert_array_equal(middle,small[::-1].transpose(0,2,1))
    assert int(survived.sum())==3*a*h
    return {"a":a,"h":h,"input_terms":int(c.sum()),
            "surviving_terms":int(survived.sum()),"output_blocks":3}

def diagonal_growth(a):
    """Lemma 5.1 lower bound, not a computable tensor-character profile P."""
    if a<1:
        raise ValueError("a>=1")
    log_H = sum(math.log1p(1/(3*m)) for m in range(1,a))
    lower = (3*a-1)/2*math.exp(log_H)
    return {"a":a,"D_lower_bound":lower,"a_power_4_over_3":a**(4/3)}

def main():
    rng = np.random.default_rng(20261009)
    max_fourier_error = 0.0
    for M in range(1,7):
        coefficients = rng.normal(size=(M,2,2,2))
        zero,error = separation(M,0.0,coefficients)
        max_fourier_error = max(max_fourier_error,error)
        expected = np.zeros_like(zero)
        for hi in range(M):
            for ui in range(M):
                expected[hi,:,hi,:,ui,:,ui] = coefficients[hi]
        np.testing.assert_array_equal(zero,expected)
    coefficients = rng.normal(size=(3,2,2,2))
    target,_ = separation(3,0.0,coefficients)
    decay = []
    for eps in (0.5,0.25,0.125,0.0625):
        transformed,_ = separation(3,eps,coefficients)
        decay.append({"epsilon":eps,"distance_to_limit":float(np.linalg.norm(transformed-target))})
    ratios = [decay[i]["distance_to_limit"]/decay[i+1]["distance_to_limit"] for i in range(3)]
    np.testing.assert_allclose(ratios,2,atol=1e-12)
    filters = [determinant_filtration(a,b) for a in range(1,6) for b in range(2,7)]
    sectors = [three_sectors(a,h) for a in range(1,7) for h in range(1,6)]
    growth = [diagonal_growth(a) for a in (1,2,3,10,100,1000)]
    assert all(v["D_lower_bound"] >= v["a_power_4_over_3"]*(1-1e-12) for v in growth)
    report = {"status":"passed","scope":"finite identities and numerical checks, not proof of full theorem",
              "separation_cases":6,"fourier_max_error":max_fourier_error,
              "epsilon_decay":decay,"determinant_filtration_cases":filters,
              "three_sector_cases":sectors,"diagonal_growth":growth,
              "matrix_algorithm_status":"Explicit rank certificate absent from paper"}
    output = Path(__file__).with_name("results")
    output.mkdir(exist_ok=True)
    (output/"paper_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("determinant_filtration_cases","three_sector_cases")},indent=2))
    print("Filtration cases:",len(filters),"Three-sector cases:",len(sectors))

if __name__=="__main__":
    main()

