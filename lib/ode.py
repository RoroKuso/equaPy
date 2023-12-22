from typing import Dict, List
from sympy import symbols, Symbol

import sympy as sp

class ODE:
    """General class to represent ODE or ODE systems"""
    
    def __init__(self, dim, order, t0, tn=None):
        self._dim = dim
        self._order = order
        self._t0 = t0
        if tn is not None:
            self._tn = tn
        self._expr = {}
            
    def setinit(self, i_values):
        self._initialValues = i_values
        
    def setsymbols(self, symbol):
        self._symb = symbol
        self._str2symb = {}
        self._str2symb["t"] = sp.symbols("t")
        for d in range(1, self._dim + 1):
            for n_deriv in range(self._order):
                tmp =  symbol + str(d) + "'" * n_deriv
                self._str2symb[tmp] = sp.symbols(tmp)
                    
    def setfunction(self, i, f):
        expr = sp.sympify(f, locals=self._str2symb)
        self._expr[i] = expr