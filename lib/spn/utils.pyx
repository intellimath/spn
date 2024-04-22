# coding: utf-8

# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: nonecheck=True
# cython: embedsignature=True
# cython: initializedcheck=True

cdef extern from "Python.h":
    void Py_INCREF(object)
    void PyTuple_SET_ITEM(object, int, object)
    void PyTuple_SetItem(object, int, object)
    object PyTuple_New(int)

import numpy as np

# cdef bint is_nge(mindex, double[::1] xj, double[::1] xk):
#     cdef int i
#     for i in mindex:
#         if xj[i] < xk[i]:
#             return 0
#     return 1

cdef bint _is_all_nge(mindex, double[::1] x0, double[:,::1] X):
    cdef int i, j, k, m
    cdef int[::1] indices
    cdef int N = X.shape[0]
    cdef double *xk
    cdef bint b_is_nge

    indices = np.array(mindex, 'i')
    m = indices.shape[0]
    if m == 0:
        return 1    
    for k in range(N):
        xk = &X[k,0]
        b_is_nge = 0
        for j in range(m):
            i = indices[j]
            if xk[i] < x0[i]:
                b_is_nge = 1
                break
        if not b_is_nge:
            return 0
    return 1

def is_all_nge(mindex, double[::1] x0, double[:,::1] X):
    return bool(_is_all_nge(mindex, x0, X))

cdef object create_submindex(tuple mindex, int j):
    cdef int i, n = len(mindex)
    
    submindex = PyTuple_New(n-1)
    
    for i in range(n):
        if i < j:
            ob = mindex[i]
            PyTuple_SET_ITEM(submindex, i, ob)
            Py_INCREF(ob)
        elif i > j:
            ob = mindex[i]
            PyTuple_SET_ITEM(submindex, i-1, ob)
            Py_INCREF(ob)
        
    return submindex

def generate_significant_submindices(mindex, double[::1] xk, double[:, ::1] Xk, seen):
    mindices = set()
    n = len(mindex)
    if n == 1:
        return mindices
    for i in range(n):
        mi = create_submindex(mindex, i)
        if mi in seen:
            continue
        if _is_all_nge(mi, xk, Xk):
            yield mi
        seen.add(mi)

def generate_significant_submindex(mindex, double[::1] xk, double[:, ::1] Xk): 
    n = len(mindex)
    if n == 1:
        return None
    for i in range(n):
        mi = create_submindex(mindex, i)
        if _is_all_nge(mi, xk, Xk):
            return mi
