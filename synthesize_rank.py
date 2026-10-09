# SPDX-License-Identifier: Apache-2.0
# Developed with OpenAI Codex assistance; see NOTICE.
"""Exact algebraic rank certificates and a fair, exhaustive synthesizer.

The paper's Remark 5.2 permits algebraic coefficients. A common number field
can be written Q[t]/(f). Enumerate monic rational f and rational coefficient
vectors, then check the tensor identity exactly. Reducible f is allowed:
an identity modulo f remains valid after evaluation at any complex root.

Conditional on the paper's existence theorem, unlimited fair search eventually
finds a certificate for any positive rational epsilon. No practical bound on
search time or usable 9/4 certificate is claimed. Defaults are bounded.
"""
from fractions import Fraction as F
from itertools import combinations, count
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np

class RationalAlgebra:
    def __init__(self, polynomial):
        self.f = tuple(F(x) for x in polynomial)
        if len(self.f)<2 or self.f[-1]!=1:
            raise ValueError("Expected a nonconstant monic rational polynomial")
        self.degree = len(self.f)-1
        self.zero = (F(0),)*self.degree
        self.one = (F(1),)+(F(0),)*(self.degree-1)

    def element(self, coefficients):
        if len(coefficients)!=self.degree:
            raise ValueError("Wrong algebra element length")
        return tuple(F(x) for x in coefficients)

    def add(self,a,b):
        return tuple(x+y for x,y in zip(a,b))

    def scale(self,a,b):
        return tuple(F(b)*x for x in a)

    def mul(self,a,b):
        d = self.degree
        c = [F(0)]*(2*d-1)
        for i,x in enumerate(a):
            for j,y in enumerate(b):
                c[i+j] += x*y
        for k in range(2*d-2,d-1,-1):
            value = c[k]
            if value:
                for j in range(d):
                    c[k-d+j] -= value*self.f[j]
                c[k] = F(0)
        return tuple(c[:d])

class ExactCertificate:
    def __init__(self,u,r,algebra,U,V,W):
        self.u,self.r,self.algebra = u,r,algebra
        if u<1 or r<1 or any(len(x)!=r*u*u for x in (U,V,W)):
            raise ValueError("Coefficient arrays must each contain r*u*u elements")
        self.U = tuple(algebra.element(x) for x in U)
        self.V = tuple(algebra.element(x) for x in V)
        # W is flattened with axes (i,j,s), unlike U,V=(s,i,j).
        self.W = tuple(algebra.element(x) for x in W)

    def verify(self):
        u,r,K = self.u,self.r,self.algebra
        for a in range(u):
            for b in range(u):
                for c in range(u):
                    for d in range(u):
                        products = [K.mul(self.U[s*u*u+a*u+b],
                                          self.V[s*u*u+c*u+d]) for s in range(r)]
                        for i in range(u):
                            for j in range(u):
                                value = K.zero
                                for s,p in enumerate(products):
                                    w = self.W[(i*u+j)*r+s]
                                    if w!=K.zero and p!=K.zero:
                                        value = K.add(value,K.mul(w,p))
                                expected = K.one if i==a and b==c and j==d else K.zero
                                if value!=expected:
                                    return False
        return True

    def qualifies(self,epsilon):
        epsilon = F(epsilon)
        if epsilon<=0 or self.u<2:
            return False
        target = F(9,4)+epsilon/2
        return self.r**target.denominator < self.u**target.numerator

    def to_dict(self):
        return {"u":self.u,"r":self.r,
                "field_polynomial":[str(x) for x in self.algebra.f],
                "U":[[str(x) for x in v] for v in self.U],
                "V":[[str(x) for x in v] for v in self.V],
                "W":[[str(x) for x in v] for v in self.W]}

    @classmethod
    def load(cls,path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(d["u"],d["r"],RationalAlgebra(d["field_polynomial"]),
                     d["U"],d["V"],d["W"])
        if not result.verify():
            raise ValueError("Exact certificate failed tensor identity")
        return result

    def numerical(self):
        # This is an approximate execution adapter, not exact certification.
        from bilinear_scheme import BilinearScheme
        root = np.roots([float(x) for x in reversed(self.algebra.f)])[0]
        def convert(values):
            return np.array([sum(complex(x)*root**i for i,x in enumerate(v)) for v in values])
        scheme = BilinearScheme(convert(self.U).reshape(self.r,self.u,self.u),
                                convert(self.V).reshape(self.r,self.u,self.u),
                                convert(self.W).reshape(self.u,self.u,self.r),
                                "exact algebraic certificate (approximate adapter)")
        scheme.verify()
        return scheme

    def multiply_exact(self,a,b):
        """Exact recursion on rational matrix inputs; returns rational entries.
        Intermediate scalars belong to Q[t]/f, not floating-point complex128.
        """
        if self.u<2:
            raise ValueError("Recursive block size must be at least 2")
        a,b = [list(row) for row in a],[list(row) for row in b]
        n = len(a)
        if len(b)!=n or any(len(row)!=n for row in a+b):
            raise ValueError("Expected equally sized square matrices")
        if n==0:
            return []
        K = self.algebra
        size = 1
        while size<n:
            size *= self.u
        aa = [[K.zero for _ in range(size)] for _ in range(size)]
        bb = [[K.zero for _ in range(size)] for _ in range(size)]
        for i in range(n):
            for j in range(n):
                aa[i][j] = K.scale(K.one,F(a[i][j]))
                bb[i][j] = K.scale(K.one,F(b[i][j]))

        def recurse(left,right):
            N = len(left)
            if N==1:
                return [[K.mul(left[0][0],right[0][0])]]
            h = N//self.u
            products = []
            for s in range(self.r):
                L = [[K.zero for _ in range(h)] for _ in range(h)]
                R = [[K.zero for _ in range(h)] for _ in range(h)]
                for i in range(self.u):
                    for j in range(self.u):
                        alpha,beta = self.U[s*self.u*self.u+i*self.u+j],self.V[s*self.u*self.u+i*self.u+j]
                        for x in range(h):
                            for y in range(h):
                                if alpha!=K.zero:
                                    L[x][y] = K.add(L[x][y],K.mul(alpha,left[i*h+x][j*h+y]))
                                if beta!=K.zero:
                                    R[x][y] = K.add(R[x][y],K.mul(beta,right[i*h+x][j*h+y]))
                products.append(recurse(L,R))
            result = [[K.zero for _ in range(N)] for _ in range(N)]
            for i in range(self.u):
                for j in range(self.u):
                    for s in range(self.r):
                        w = self.W[(i*self.u+j)*self.r+s]
                        if w!=K.zero:
                            for x in range(h):
                                for y in range(h):
                                    result[i*h+x][j*h+y] = K.add(result[i*h+x][j*h+y],K.mul(w,products[s][x][y]))
            return result
        output = recurse(aa,bb)
        assert all(all(v==0 for v in output[i][j][1:]) for i in range(n) for j in range(n))
        return [[output[i][j][0] for j in range(n)] for i in range(n)]

def rational_at(index):
    """Unique enumeration of Q: 0, then signed Calkin-Wilf rationals."""
    if index<0:
        raise ValueError("index must be nonnegative")
    if index==0:
        return F(0)
    n = (index+1)//2
    a,b = 1,1
    for bit in bin(n)[3:]:
        a,b = (a,a+b) if bit=="0" else (a+b,b)
    return F(a,b) if index%2 else F(-a,b)

def compositions(total,length):
    """Iterative weak compositions; no recursion-depth limit."""
    for bars in combinations(range(total+length-1),length-1):
        previous = -1
        values = []
        for bar in bars:
            values.append(bar-previous-1)
            previous = bar
        values.append(total+length-2-previous)
        yield tuple(values)

def rational_vectors(length):
    for shell in count(0):
        for indices in compositions(shell,length):
            yield tuple(rational_at(i) for i in indices)

def candidates(u,r,K):
    chunk = r*u*u
    for vector in rational_vectors(3*chunk*K.degree):
        values = [vector[i:i+K.degree] for i in range(0,len(vector),K.degree)]
        yield ExactCertificate(u,r,K,values[:chunk],values[chunk:2*chunk],values[2*chunk:])

def tasks(epsilon):
    target = F(9,4)+F(epsilon)/2
    if target<=F(9,4):
        raise ValueError("epsilon must be positive")
    for shell in count(3):
        for u in range(2,shell):
            valid_ranks = [r for r in range(u*u,u**3+1)
                           if r**target.denominator < u**target.numerator]
            for degree in range(1,shell-u+1):
                for indices in compositions(shell-u-degree,degree):
                    K = RationalAlgebra([rational_at(i) for i in indices]+[F(1)])
                    for r in valid_ranks:
                        yield (u,r,K)

def fair_search(epsilon,max_rounds=40,max_seconds=5.0):
    """One new task per round; advance every active coefficient enumeration.
    Each finite task and coefficient vector eventually gets visited in unlimited
    mode. Budgets are checked between candidates, not within one exact check.
    """
    epsilon = F(epsilon)
    if epsilon<=0:
        raise ValueError("epsilon must be positive")
    start,checked = time.monotonic(),0
    schedule,active = tasks(epsilon),[]
    for round_index in count(1):
        if max_rounds is not None and round_index>max_rounds:
            break
        if max_seconds is not None and time.monotonic()-start>=max_seconds:
            break
        u,r,K = next(schedule)
        active.append(candidates(u,r,K))
        for iterator in active:
            if max_seconds is not None and time.monotonic()-start>=max_seconds:
                return None,{"status":"budget_exhausted","rounds":round_index,
                             "candidates_checked":checked,"seconds":time.monotonic()-start}
            certificate = next(iterator)
            checked += 1
            if certificate.verify():
                assert certificate.qualifies(epsilon)
                return certificate,{"status":"certificate_found","rounds":round_index,
                                    "candidates_checked":checked,"seconds":time.monotonic()-start}
    return None,{"status":"budget_exhausted","rounds":round_index-1,
                 "candidates_checked":checked,"seconds":time.monotonic()-start}

def strassen_exact(algebraic=False):
    """Known Strassen witness only; not a newly synthesized 9/4 certificate."""
    from bilinear_scheme import strassen_certificate
    s = strassen_certificate()
    K = RationalAlgebra([1,0,1] if algebraic else [0,1])
    arrays = []
    for arr in (s.U,s.V,s.W):
        arrays.append([K.scale(K.one,int(x.real)) for x in arr.flat])
    if algebraic:
        t = (F(0),F(1))
        arrays[0] = [K.mul(v,t) for v in arrays[0]]
        arrays[1] = [K.mul(v,K.scale(t,-1)) for v in arrays[1]]
    return ExactCertificate(2,7,K,*arrays)

def demo():
    checks = 0
    assert len({rational_at(i) for i in range(100)})==100
    for total,length in ((0,3),(1,3),(4,3)):
        vectors = list(compositions(total,length))
        assert len(vectors)==math.comb(total+length-1,length-1)
        assert all(sum(v)==total and len(v)==length for v in vectors)
        checks += 1
    K = RationalAlgebra([1,0,1])
    t = (F(0),F(1))
    assert K.mul(t,t)==K.scale(K.one,-1)
    # Actually synthesize scalar rank one, to test enumeration/verification.
    scalar_checked = 0
    for c in candidates(1,1,RationalAlgebra([0,1])):
        scalar_checked += 1
        if c.verify():
            break
        assert scalar_checked<1000
    for algebraic in (False,True):
        c = strassen_exact(algebraic)
        assert c.verify() and not c.qualifies("1/10") and c.qualifies("6/5")
        for n in (1,2,3,4):
            a = [[F(i-j,3) for j in range(n)] for i in range(n)]
            b = [[F(i+j+1,5) for j in range(n)] for i in range(n)]
            expected = [[sum(a[i][k]*b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            assert c.multiply_exact(a,b)==expected
            checks += 1
        c.numerical()
        checks += 1
    corrupted = strassen_exact()
    corrupted.U = (corrupted.algebra.zero,)+corrupted.U[1:]
    assert not corrupted.verify()
    checks += 1
    cert,search_report = fair_search("1/10",max_rounds=20,max_seconds=2)
    assert cert is None
    report = {"exact_tests_passed":checks,"scalar_rank_one_candidates":scalar_checked,
              "bounded_search_epsilon":"1/10","bounded_search":search_report,
              "fast_certificate_found":False,
              "scope":"search mechanism and exact Strassen tests; no 9/4 runtime claim"}
    output = Path(__file__).with_name("results")
    output.mkdir(exist_ok=True)
    (output/"synthesis_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",action="store_true")
    parser.add_argument("--epsilon",default="1/10")
    parser.add_argument("--max-rounds",type=int,default=40)
    parser.add_argument("--max-seconds",type=float,default=5)
    parser.add_argument("--unbounded",action="store_true",help="Disable search budgets; may never finish in practical time")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("results")/"synthesized_certificate.json")
    args = parser.parse_args()
    if args.demo:
        demo()
        return
    if args.max_rounds<1 or args.max_seconds<=0:
        parser.error("budgets must be positive")
    certificate,report = fair_search(args.epsilon,
        None if args.unbounded else args.max_rounds,
        None if args.unbounded else args.max_seconds)
    print(json.dumps(report,indent=2),flush=True)
    if certificate is None:
        print("No admissible rank certificate found. No 9/4 algorithm is available for benchmarking.")
        return
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(certificate.to_dict(),indent=2),encoding="utf-8")
    print("Exact certificate saved:",args.output)

if __name__=="__main__":
    main()

