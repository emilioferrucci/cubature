import numpy as np
import scipy.linalg
import functools, sys, os, math, itertools, operator, six
from sympy.utilities.iterables import multiset_permutations, ordered_partitions, kbins
from sympy.ntheory import mobius, divisors
from sympy import Rational
import sympy
import unittest
import pyparsing
from free_lie_algebra import *

def symConcatProdMany(a, maxLevel=None):
    f = 1/math.factorial(len(a))
    s = word2Elt(emptyWord)
    for b in list(itertools.permutations(a)): #why doesn't sum() of Elts work
        s = s + concatenationProductMany(b)
    s = s - word2Elt(emptyWord)
    return f*s

def Psym(w, basis):
    assert isinstance(basis, HallBasis), basis
    assert type(w) in (tuple,str), w
    if 0==len(w):
        return unitElt
    assert 0<len(w)<=basis.m
    a=basis.factorIntoHallWords(w)
    h = [basisElementToElt(i) for i in a]
    out = symConcatProdMany(h)
    return out

def strip_empties(ee):
    for k in list(ee.keys()):
        if any(w == Word([]) for w in k):
            del ee[k]
    return ee

def delta_reduced(a,p=2):
    return EltElt(strip_empties(delta(a,p).data),p)

def dict_word_to_elt(dict):
     return {tuple(word2Elt(item) for item in key): value for key, value in dict.items()}

def list_unshuffle(ee, max_deg): # in the future it would be better to get max_deg from the input; a larger max_deg won't hurt
    list = []
    for n in range(1, max_deg+1):
        list.append(dict_word_to_elt(delta_reduced(ee,n).data))
    return list #a list of dicts, each corresponding to an degree-n unshuffle; each dict is has coeffs as values and tuples of elts as keys

def pi_tuple(tup):
    return tuple(pi1(x) for x in tup)

def removeSmalls(a):
    """a version of an Elt with tiny elements removed"""
    assert isinstance(a,Elt), a
    d=[{k:v for k,v in i.items() if math.fabs(v)>1e-4} for i in a.data] # I chose 4 based on examples
    return Elt(d)