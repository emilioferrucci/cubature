# to use cpu uncomment the following:
#import os
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roughpy as rp
import math
import jax
import jax.numpy as jnp
from diffrax import *
import time

# The following is meant to take a function f, interpolate it on N points between s and t, and compute it signature in roughpy

def interp_f(f, N, s, t):
    x_known = np.linspace(s, t, N)
    y_known = np.array([f(x) for x in x_known])
    def interpolated_f(x_interp):
        y_interp = np.zeros(y_known.shape[1])
        for i in range(y_known.shape[1]):
            y_interp[i] = np.interp(x_interp, x_known, y_known[:, i])
        return y_interp
    return interpolated_f

def roughpy_bm(key, N, d, s = 0, t = 1):
    vbt = VirtualBrownianTree(s, t, tol=1/N, shape=(d,), key=key)
    bm = interp_f(vbt.evaluate, N, s, t)
    return lambda t, ctx: rp.Lie(bm(t), ctx=ctx)

def make_brownian_sig(key, N, res, d, n, s = 0, t = 1):
    rpbm = roughpy_bm(key, N, d, s, t)
    context = rp.get_context(width = d, depth = n, coeffs=rp.DPReal)
    function_stream = rp.FunctionStream.from_function(rpbm, ctx = context, resolution = res)
    return function_stream.signature(rp.RealInterval(s,t))

def _sig_degrees(sig, d, n):
    expected_length = (d ** (n + 1) - 1)/(d - 1)
    assert len(sig) == expected_length, f"Array length must be {expected_length}, but got {len(sig)}"
    result = []
    start = 0
    for i in range(n + 1):
        length = d ** i
        subarray = sig[start:start + length]
        result.append(subarray)
        start += length
    return result

def _reshape_level(arr, d, n):
    expected_length = d ** n
    assert len(arr) == expected_length, f"Array length must be {expected_length}, but got {len(arr)}"
    
    new_shape = (d,) * n
    reshaped_array = arr.reshape(new_shape)
    return reshaped_array

def reshape_signature(sig, d, n): 
    npsig = np.array(sig)
    result = []
    k = 0
    for arr in _sig_degrees(npsig, d, n):
        result.append(jnp.array(_reshape_level(arr, d, k)))
        k += 1
    return result


# Computes the powers of the tensor A up to order n and stores them in a list of jnp arrays.

def powers_up_to(A, n):
    matrix = A
    result = [matrix]
    for i in range(1, n):
        subscripts = 'ab' + ''.join(chr(100 + k) for k in range(i)) + ',bc' + chr(100 + i) + '->ac' + ''.join(chr(100 + k) for k in range(i + 1))
        matrix = jnp.einsum(subscripts, matrix, A)
        result.append(matrix)
    return result


# Computes the one-step Euler approximation of the linear CDE with given signature.

def make_indices(n):
    indices_A = 'ab' + ''.join(chr(99 + i) for i in range(n-1, -1, -1))
    indices_S = ''.join(chr(99 + i) for i in range(n))
    output_indices = 'ab'
    return indices_A, indices_S, output_indices

def single_sum_euler(n, y0, An, Sn, indices_list):
    R = jnp.einsum(f'{indices_list[0]},{indices_list[1]}->{indices_list[2]}', An, Sn)
    return jnp.dot(R, y0)

def one_step_euler(n, y0, powers, S, indices_list):
    return y0 + sum(_single_sum_euler(k, y0, powers[k], S[k+1], indices_list[k]) for k in range(n-1))