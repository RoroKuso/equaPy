"""
Contains algorithms for numerical resolution of ODEs 
"""

import numpy as np
import sympy as sp
import torch
import copy

from lib.ode import ODE
from math import factorial
from lib.constants import RKdict

def _newton(f, y_n, treshold=1e-7):
    n = len(y_n)
    inputs = torch.from_numpy(y_n)
    jac = torch.autograd.functional.jacobian(f, inputs)
    b = torch.diag(torch.ones(n, dtype=torch.double))
    
    if not torch.linalg.det(jac):
        return y_n
    
    inv_jac = torch.linalg.solve(jac, b)
    next_inputs = inputs - torch.matmul(inv_jac, f(inputs))
    
    while torch.linalg.vector_norm(next_inputs - inputs) > treshold:
        jac = torch.autograd.functional.jacobian(f, next_inputs)
        if not torch.linalg.det(jac):
            break
        inv_jac = torch.linalg.solve(jac, b)
        inputs = next_inputs
        next_inputs = inputs - torch.matmul(inv_jac, f(inputs))
    
    return next_inputs.detach().numpy()

def _RKCheck(butcher):
    """
    Return True if the butcher table is explicit. False otherwise
    """
    q = len(butcher)
    for i in range(q):
        for j in range(i, q):
            if butcher[i][j] != 0:
                return False
    return True

class Scheme:
    def __init__(self):
        raise NotImplementedError("Can't create instance of interface class Scheme !")
    
    def _next_value(self, time, value, h):
        return value
    
    def _step_update(self, time, value, h):
        """
        Returns the next value of the scheme with some additional behavior (update layer, printing...).
        """
        v = self._next_value(time, value, h)
        for f in self._ode._update_layer:
            f(v)
        return v
    
    def solve(self, T, N):
        """
        General method to solve ODEs.
        
        Parameters
        ----------
        T : float
            Time limit.
        N : int
            Number of points.
        """
        
        # Initialization
        ode: ODE = self._ode
        n = ode._dim
        h: float = T / N
        time = np.array([ode._t0 + i * h for i in range(N)])
        value = np.empty((N, n * ode._order))
        value[0] = np.array(ode._initialValues)
        
        # Loop method
        for i in range(1, N):
            if not i%100:
                print(f"step {i}")
            v = self._step_update(time[i-1], value[i-1], h)
            value[i] = v
            
        # Process result
        ret = np.empty((N, ode._dim))
        for i in range(N):
            ret[i] = value[i][:ode._dim]

        return (time, ret)
    
class ExplicitEuler(Scheme):
    name = "Explicit Euler"
    def __init__(self, ode: ODE):
        self._ode = ode
        
    def _next_value(self, t_n, y_n, h):
        flam = self._ode._mainflam
        tmp = np.array(flam(t_n, *y_n))
        return y_n + h * tmp   
    
class ModifiedEuler(Scheme):
    name = "Modified Euler"
    def __init__(self, ode: ODE):
        self._ode = ode
        self._ode.set_linearized_flam()
        
    def _next_value(self, t_n, y_n, h):
        flam = self._ode._mainflam
        tmp = y_n + (h/2) * np.array(flam(t_n + h/2, *y_n))
        ret = np.array(flam(t_n + h/2, *tmp))
        return y_n + h * ret
    
class ImplicitEuler(Scheme):
    name = "Implicit Euler"
    def __init__(self, ode: ODE):
        self._ode = ode
        self._ode.set_linearized_flam()
         
    def _next_value(self, t_n, y_n, h):
        tmp = self._ode._mainflam
        def _to_solve(x):
            arg = x.detach().numpy()
            return x - h * torch.from_numpy(np.array(tmp(t_n+h, *arg))) - torch.from_numpy(y_n)
        return _newton(_to_solve, y_n)   
    
class CrankNicolson(Scheme):
    name = "Crank Nicolson"
    def __init__(self, ode: ODE):
        self._ode = ode
        self._ode.set_linearized_flam()
        
    def _next_value(self, t_n, y_n, h):
        tmp = self._ode._mainflam
        
        def _to_solve(x):
            arg = x.detach().numpy()
            return x - torch.from_numpy(y_n) - (h/2) * (
                torch.from_numpy(np.array(tmp(t_n, *y_n))) - \
                torch.from_numpy(np.array(tmp(t_n+h, *arg))))
            
        return _newton(_to_solve, y_n)
    
class Taylor(Scheme):
    def __init__(self, ode: ODE, p: int = 1):
        self._ode = ode
        self._order = p
        self._derF = []
        self.name = f"Taylor p = {p}"
        self._ode.set_linearized_flam()
        
    def _process_derivatives(self):
        ode = self._ode
        symbols = np.array([ode._str2symb[e] for e in ode._str2symb])
        
        tmp = symbols[1+ode._dim:]
        for i in range(1, ode._dim+1):
            tmp = np.append(tmp, ode._expr[i])
            
        self._derF.append(tmp)
        for k in range(1, self._order+1):
            deriv1 = np.array([])
            for i in range(ode._dim * ode._order):
                deriv1 = np.append(deriv1, sp.diff(self._derF[k-1][i], ode._str2symb["t"]))
            
            for j in range(ode._dim * ode._order):
                deriv2 = np.array([])

                for i in range(ode._dim * ode._order):
                    deriv2 = np.append(deriv2, sp.diff(self._derF[k-1][i], symbols[1+j]) * self._derF[0][j])
                    
                deriv1 += deriv2
            
            self._derF.append(deriv1)
     
    def _computeF(self, h):
        symbols = np.array([self._ode._str2symb[e] for e in self._ode._str2symb])
        res = self._derF[0]
        for k in range(2, self._order+1):
            res += self._derF[k-1] * (h**(k-1)) / factorial(k)    
        self._Flam = sp.lambdify(symbols, list(res))
        
    def _next_value(self, t_n, y_n, h):
        return y_n + h * np.array(self._Flam(t_n, *y_n))
    
    def solve(self, T, N):
        h = T / N
        self._process_derivatives()
        self._computeF(h)
        return super().solve(T, N)
    
class RungeKutta(Scheme):
    def __init__(self, ode: ODE, butcher="RK4"):
        self._ode = ode
        self._ode.set_linearized_flam()
        if isinstance(butcher, str):
            self._butcher = RKdict[butcher]
            self.name = f"Runge-Kutta {butcher}"
        else:
            self._butcher = copy.deepcopy(butcher)
            self.name = "Runge-Kutta (custom butcher)"
            
        self._explicit = _RKCheck(self._butcher[0])
        self._q = len(self._butcher[0])
        
    def _next_value(self, t_n, y_n, h):
        return self._next_value_explicit(t_n, y_n, h) if self._explicit else self._next_value_implicit(t_n, y_n, h)
    
    def _next_value_explicit(self, t_n, y_n, h):
        q = self._q
        flam = self._ode._mainflam
        yi_val = [copy.deepcopy(y_n)]
        
        for i in range(1, q):
            tmp = copy.deepcopy(y_n)
            for j in range(i):
                tmp += self._butcher[0][i][j] * h * np.array(flam(t_n + self._butcher[2][j] * h, *yi_val[j]))
            yi_val.append(tmp)
            
        res = copy.deepcopy(y_n)
        for i in range(q):
            res += self._butcher[1][i] * h * np.array(flam(t_n + self._butcher[2][i]*h, *yi_val[i]))
            
        return res
            
    def _next_value_implicit(self, t_n, y_n, h):
        ode = self._ode
        q = self._q
        flam = ode._mainflam
        
        def _to_solve(x):
            arg = x.detach().numpy()
            yi = np.array([])
            for i in range(q):
                tmp = copy.deepcopy(y_n)
                for j in range(q):
                    start = ode._dim * ode._order * j
                    end = ode._dim * ode._order * (j + 1)
                    tmp += self._butcher[0][i][j] * h * np.array(flam(t_n + self._butcher[2][j] * h, *arg[start:end]))
                        
                yi = np.append(yi, tmp)
            return x - torch.from_numpy(yi)
        
        v0 = np.array([])
        for i in range(q):
            v0 = np.append(v0, y_n)
        yi_val = _newton(_to_solve, v0)
        
        res = copy.deepcopy(y_n)
        for i in range(q):
            start = ode._dim * ode._order * i
            end = ode._dim * ode._order * (i + 1)
            res += self._butcher[1][i] * h * np.array(flam(t_n + self._butcher[2][i]*h, *yi_val[start:end]))
        return res