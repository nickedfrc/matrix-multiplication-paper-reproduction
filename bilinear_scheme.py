# SPDX-License-Identifier: Apache-2.0
# Developed with OpenAI Codex assistance; see NOTICE for mathematical sources.
"""Generic executor for the paper's Section 5 conditional rank recursion.
U,V: (r,u,u); W: (u,u,r); C_ij=sum_s W_ijs (sum U_sab A_ab)(sum V_scd B_cd).
No 9/4 certificate is supplied. Strassen certificate is only an executor test.
"""
import math
import numpy as np
from matmul_compare import inputs

class BilinearScheme:
    def __init__(self,U,V,W,name="supplied"):
        self.U,self.V,self.W = [np.asarray(x,dtype=np.complex128) for x in (U,V,W)]
        if self.U.ndim!=3:
            raise ValueError("U must have axes r,u,u")
        self.r,self.u,last = self.U.shape
        if self.u<2 or self.r<1 or last!=self.u or self.V.shape!=self.U.shape or self.W.shape!=(self.u,self.u,self.r):
            raise ValueError("Expected U,V=(r,u,u), W=(u,u,r), u>=2")
        if not all(np.isfinite(x).all() for x in (self.U,self.V,self.W)):
            raise ValueError("Coefficients must be finite")
        self.name = name

    @property
    def exponent(self):
        return math.log(self.r,self.u)

    def verify(self):
        # Exhaustive basis-pair checks avoid a dense u**6 tensor allocation.
        worst = 0.0
        for a in range(self.u):
            for b in range(self.u):
                for c in range(self.u):
                    for d in range(self.u):
                        result = np.einsum("ijs,s->ij",self.W,self.U[:,a,b]*self.V[:,c,d])
                        expected = np.zeros((self.u,self.u))
                        if b==c:
                            expected[a,d] = 1
                        worst = max(worst,float(np.max(np.abs(result-expected))))
        if worst>1e-12:
            raise ValueError(f"Rank certificate failed: {worst}")
        return worst  # floating validation; integer Strassen coefficients are exact

    def _multiply(self,a,b,leaf):
        n = len(a)
        if n<=leaf:
            return a@b
        h = n//self.u
        aa = a.reshape(self.u,h,self.u,h).transpose(0,2,1,3)
        bb = b.reshape(self.u,h,self.u,h).transpose(0,2,1,3)
        left = np.einsum("sij,ijab->sab",self.U,aa)
        right = np.einsum("sij,ijab->sab",self.V,bb)
        products = np.stack([self._multiply(left[s],right[s],leaf) for s in range(self.r)])
        blocks = np.einsum("ijs,sab->ijab",self.W,products)
        return blocks.transpose(0,2,1,3).reshape(n,n)

    def multiply(self,a,b,leaf_size=128):
        if not isinstance(leaf_size,int) or leaf_size<1:
            raise ValueError("leaf_size must be a positive integer")
        a,b = inputs(a,b)
        if a.shape!=b.shape or a.shape[0]!=a.shape[1]:
            raise ValueError("Generic executor requires equally sized square matrices")
        n = len(a)
        if n==0:
            return np.zeros_like(a,dtype=np.complex128)
        size = 1
        while size<n:
            size *= self.u
        aa,bb = np.zeros((size,size),complex),np.zeros((size,size),complex)
        aa[:n,:n],bb[:n,:n] = a,b
        return self._multiply(aa,bb,leaf_size)[:n,:n].copy()

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            scheme = cls(data["U"],data["V"],data["W"],str(path))
        scheme.verify()
        return scheme

def strassen_certificate():
    U,V,W = np.zeros((7,2,2)),np.zeros((7,2,2)),np.zeros((2,2,7))
    U[0,0,0]=U[0,1,1]=1
    U[1,1,0]=U[1,1,1]=1
    U[2,0,0]=U[3,1,1]=1
    U[4,0,0]=U[4,0,1]=1
    U[5,1,0]=1; U[5,0,0]=-1
    U[6,0,1]=1; U[6,1,1]=-1
    V[0,0,0]=V[0,1,1]=1
    V[1,0,0]=1
    V[2,0,1]=1; V[2,1,1]=-1
    V[3,1,0]=1; V[3,0,0]=-1
    V[4,1,1]=1
    V[5,0,0]=V[5,0,1]=1
    V[6,1,0]=V[6,1,1]=1
    W[0,0,[0,3,4,6]]=[1,1,-1,1]
    W[0,1,[2,4]]=1
    W[1,0,[1,3]]=1
    W[1,1,[0,1,2,5]]=[1,-1,1,1]
    return BilinearScheme(U,V,W,"Strassen")

if __name__=="__main__":
    scheme = strassen_certificate()
    error = scheme.verify()
    rng = np.random.default_rng(20261009)
    for n in (1,2,3,5,8,17):
        a,b = rng.normal(size=(n,n)),rng.normal(size=(n,n))
        np.testing.assert_allclose(scheme.multiply(a,b,1),a@b,rtol=1e-10,atol=1e-10)
    bad = strassen_certificate()
    bad.U[0,0,0] = 2
    try:
        bad.verify()
    except ValueError:
        pass
    else:
        raise AssertionError("Bad certificate accepted")
    print("Generic executor passed; Strassen certificate error:",error,"exponent:",scheme.exponent)

