import numpy as np
import sympy as sp
import copy

from lib.ode import ODE
from typing import Optional
from math import factorial

from lib.constants import RKdict

def newton_n(f, df, x0, context, eps=1e-10):
    if df.det() == 0:
        return x0
    
    dfinv = df.inv()
    x_n = x0
    tmp = dfinv * f
    tmpflam = sp.lambdify(context, tmp, "numpy")
    x_next = x_n - np.array(tmpflam(*x_n)).flatten()
    
    while np.linalg.norm(x_next - x_n) > eps:
        x_n = x_next
        x_next = x_n - np.array(tmpflam(*x_n)).flatten()
        
    return x_next

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
    
    def solve(self, T, N):
        ode = self._ode
        n = ode._dim
        h: float = T / N
        time = np.array([ode._t0 + i * h for i in range(N)])
        value = np.zeros((N, n * ode._order))
        value[0] = ode._initialValues
        for i in range(1, N):
            v = self._next_value(time[i-1], value[i-1], h)
            value[i] = v
        res = np.array([v[:self._ode._dim] for v in value])
        return (time, res)
    
class ExplicitEuler(Scheme):
    name = "Explicit Euler"
    def __init__(self, ode: ODE):
        self._ode = ode
         
    def _next_value(self, t_n, y_n, h):
        ode = self._ode
        symbols = [ode._str2symb[e] for e in ode._str2symb]
        tmp = y_n[ode._dim::]
        for i in range(1, self._ode._dim+1):
            flam = sp.lambdify(symbols, ode._expr[i])
            tmp = np.append(tmp, flam(t_n, *y_n))
        return y_n + h * tmp
            
    
class ModifiedEuler(Scheme):
    name = "Modified Euler"
    def __init__(self, ode: ODE):
        self._ode = ode
        
    def _next_value(self, t_n, y_n, h):
        ode = self._ode
        symbols = [ode._str2symb[e] for e in ode._str2symb]
        tmp = y_n[ode._dim::]
        for i in range(1, ode._dim+1):
            flam = sp.lambdify(symbols, ode._expr[i])
            new_y = y_n + (h/2) * flam(t_n, *y_n)
            tmp = np.append(tmp, flam(t_n + h/2, *new_y))
        return y_n + h * tmp
        
            
class ImplicitEuler(Scheme):
    name = "Implicit Euler"
    def __init__(self, ode: ODE):
        self._ode = ode
        
    def _next_value(self, t_n, y_n, h):
        ode = self._ode

        y_symbols = np.array([ode._str2symb[e] for e in ode._str2symb if e != "t"])
        t = ode._str2symb["t"]
        tmp = y_n[ode._dim::]
        for i in range(1, ode._dim+1):
            tmp = np.append(tmp, ode._expr[i].subs(t, t_n+h))
        
        to_solve = y_symbols - y_n - h * tmp
        V = sp.Matrix(to_solve)
        J = sp.Matrix.jacobian(V, y_symbols)
        return newton_n(V, J, y_n, y_symbols)
    
class CrankNicolson(Scheme):
    name = "Crank Nicolson"
    def __init__(self, ode: ODE):
        self._ode = ode
        
    def _next_value(self, t_n, y_n, h):
        ode = self._ode

        symbols = [ode._str2symb[e] for e in ode._str2symb]
        y_symbols = np.array([ode._str2symb[e] for e in ode._str2symb if e != "t"])
        t = ode._str2symb["t"]
        
        tmp1 = y_n[ode._dim::]
        tmp2 = y_n[ode._dim::]
        
        for i in range(1, ode._dim+1):
            flam = sp.lambdify(symbols, ode._expr[i])
            tmp1 = np.append(tmp1, flam(t_n, *y_n))
            tmp2 = np.append(tmp2, ode._expr[i].subs(t, t_n+h))
        
        to_solve = y_symbols - y_n - (h/2) * (tmp1 + tmp2)
        V = sp.Matrix(to_solve)
        J = sp.Matrix.jacobian(V, y_symbols)
        return newton_n(V, J, y_n, y_symbols)
    
class Taylor(Scheme):
    def __init__(self, ode: ODE, p: int = 1):
        self._ode = ode
        self._order = p
        self._derF = {}
        self.name = f"Taylor p = {p}"
        
    def _process_derivatives(self):
        ode = self._ode
        symbols = np.array([ode._str2symb[e] for e in ode._str2symb])
        y_symbols1 = np.array([e for e in ode._str2symb if e != "t"])
        y_symbols2 = np.array([ode._str2symb[e] for e in ode._str2symb if e != "t"])
        tmp = y_symbols2[ode._dim::]
        for i in range(1, ode._dim+1):
            tmp = np.append(tmp, ode._expr[i])
            
        self._derF[0] = tmp
        for k in range(1, self._order+1):
            deriv1 = np.array([])
            for i in range(ode._dim * ode._order):
                deriv1 = np.append(deriv1, sp.diff(self._derF[k-1][i], ode._str2symb["t"]))
            
            for j in range(ode._dim * ode._order):
                deriv2 = np.array([])

                for i in range(ode._dim * ode._order):
                    deriv2 = np.append(deriv2, sp.diff(self._derF[k-1][i], y_symbols2[j]))
                    deriv2[i] = deriv2[i] * self._derF[0][j]
                    
                deriv1 += deriv2
                
            self._derF[k] = deriv1
     
    def _computeF(self, h):
        res = self._derF[0]
        for k in range(2, self._order+1):
            res += self._derF[k-1] * (h**(k-1)) / factorial(k)    
        self._F = res
        
    def _next_value(self, t_n, y_n, h):
        ode = self._ode
        symbols = [ode._str2symb[e] for e in ode._str2symb]
        F = self._F
        Flam = sp.lambdify(symbols, list(F))
        
        return y_n + h * np.array(Flam(t_n, *y_n))
    
    def solve(self, T, N):
        h = T / N
        self._process_derivatives()
        self._computeF(h)
        return super().solve(T, N)
    
class RungeKutta(Scheme):
    def __init__(self, ode: ODE, butcher="RK4"):
        self._ode = ode
        if isinstance(butcher, str):
            self._butcher = RKdict[butcher]
            self.name = f"Runge-Kutta {butcher}"
        else:
            self._butcher = copy.deepcopy(butcher)
            self.name = "Runge-Kutta (custom butcher)"
            
        self._explicit = _RKCheck(self._butcher[0])
        
        
    def _next_value_explicit(self, t_n, y_n, h):
        q = len(self._butcher[0])
        symbols = [self._ode._str2symb[s] for s in self._ode._str2symb]
        y_symbols = [self._ode._str2symb[s] for s in self._ode._str2symb if s != "t"]
        yi = []
        for i in range(q):
            tot = []
            for j in range(i):
                expr = self._butcher[0][i][j] * np.array(y_symbols[self._ode._dim:])
                for k in range(1, self._ode._dim+1):
                    expr = np.append(expr, self._butcher[0][i][j] * self._ode._expr[k])
                tot.append(expr)
            yi.append(tot)
            
        yi_val = [y_n]
        for i in range(1, q):
            tmp = copy.deepcopy(y_n)
            for j in range(i):
                flam = sp.lambdify(symbols, list(yi[i][j]))
                tmp += h * np.array(flam(t_n + self._butcher[2][j] * h, *yi_val[j]))
            yi_val.append(tmp)
            
        res = copy.deepcopy(y_n)
        for i in range(q):
            flam = sp.lambdify(symbols, [self._ode._expr[j] for j in range(1, self._ode._dim+1)])
            res += self._butcher[1][i] * h * np.array(flam(t_n + self._butcher[2][i]*h, *yi_val[i]))
            
        return res
            
        
    def _solve_explicit(self, T, N):
        ode = self._ode
        n = ode._dim
        h: float = T / N
        time = np.array([ode._t0 + i * h for i in range(N)])
        value = np.zeros((N, n * ode._order))
        value[0] = ode._initialValues
        for i in range(1, N):
            v = self._next_value_explicit(time[i-1], value[i-1], h)
            value[i] = v
        res = np.array([v[:ode._dim] for v in value])
        return (time, res)
        
    def solve(self, T, N):
        if self._explicit:
            return self._solve_explicit(T, N)
        else:
            return self._solve_implicit(T, N)
            
        