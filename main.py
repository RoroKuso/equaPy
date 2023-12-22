from lib.cauchy import Cauchy
from lib.cauchy_methods import ExplicitEuler, error_comparison
from lib.ode import ODE
from lib.ode_methods import ExplicitEuler, ModifiedEuler, ImplicitEuler, CrankNicolson, Taylor, RungeKutta

import math

import sympy as sp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def example7():
    cauchy = Cauchy(1, 1, 0)
    v0 = np.zeros((1,1))
    cauchy.setinit(v0)
    cauchy.setsymbol("Y")
    cauchy.setfunction("-10 * (Y - t ** 2) + 2 * t")

    scheme = ExplicitEuler(cauchy)
    
    # X = np.linspace(0, 1, 20)
    # Y = np.array([t*t for t in X])
    # x1, y1 = scheme.solve(1, 5)
    # x2, y2 = scheme.solve(1, 20)
    
    # plt.plot(X, Y, label="exact solution")
    # plt.plot(x1, y1, label="explicit euler N = 5")
    # plt.plot(x2, y2, label="explicit euler N = 20")
    # plt.legend()
    # plt.show()
    
    def sol(t):
        return [t ** 2]
    
    h_range, error = scheme.error(sol, np.arange(5, 200, 10), 1)
    C = 1/4
    p1 = error_comparison(h_range, 1, 1)
    p2 = error_comparison(h_range, 1/10, 2)
    plt.plot(h_range, error, label="log error")
    plt.plot(h_range, p1, label=f"linear : {C} * h")
    plt.plot(h_range, p2, label=f"quadratic : {C} * h^2")
    plt.xscale('log')
    plt.yscale('log')
    
    
    plt.title("logarithmic error according to step value")
    plt.legend()
    plt.show()
    
def example8():
    "bink tank cascade"
    ode = ODE(3, 1, 0)
    ode.setinit([20, 40, 60])
    ode.setsymbols("x")
    ode.setfunction(1, "-x1/2")
    ode.setfunction(2, "x1/2 - x2/4")
    ode.setfunction(3, "x2/4 - x3/6")
    
    scheme = RungeKutta(ode, butcher="RK2")
    time, values = scheme.solve(50, 200)
    
    plt.plot(time, values[:,0], label='upper tank')
    plt.plot(time, values[:,1], label='middle tank')
    plt.plot(time, values[:,2], label='lower tank')
    plt.legend()
    plt.show()
    
def example9():
    "coupled spring-mass system"
    m1, m2, m3 = 5, 5, 5
    k1, k2, k3, k4 = 2, 1, 1, 2
    ode = ODE(3, 2, 0)
    ode.setinit([1, 0, 0, 0, 0, 0])
    ode.setsymbols("x")
    ode.setfunction(1, f"(-{k1} * x1 + {k2} * (x2 - x1)) / {m1}")
    ode.setfunction(2, f"(-{k2} * (x2 - x1) + {k3} * (x3 - x2)) / {m2}")
    ode.setfunction(3, f"(-{k3} * (x3 - x2) - {k4} * x3) / {m3}")
    
    scheme = ExplicitEuler(ode)
    time, values = scheme.solve(20, 100)
    
    print(values)
    
    fig, ax = plt.subplots()
    scat1 = ax.scatter(0, values[:,0][0], c="b", s=5, label=f'x1')
    scat2 = ax.scatter(0, values[:,1][0], c="g", s=5, label=f'x2')
    scat3 = ax.scatter(0, values[:,2][0], c="r", s=5, label=f'x3')
    
    ax.set(xlim=[-0, 20], ylim=[-5, 5], xlabel="position")
    ax.legend()
    
    def update(frame):
        y1 = values[:,0][:frame]
        y2 = values[:,1][:frame]
        y3 = values[:,2][:frame]
        x = time[:frame]
        data1 = np.stack([x, y1]).T
        scat1.set_offsets(data1)
        data2 = np.stack([x, y2]).T
        scat2.set_offsets(data2)
        data3 = np.stack([x, y3]).T
        scat3.set_offsets(data3)
        return scat1, scat2, scat3
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=10)
    plt.show()
    
def example10():
    "spring mass system (fr : masse suspendue par un ressort ressort)"
    M = 1
    m = 0.0001
    k = 1
    g = 9.81
    w = math.sqrt(k / (M + m/3))
    yeq = -(M + m/3) * g / k
    
    ode = ODE(1, 2, 0)
    ode.setinit([1, 0])
    ode.setsymbols("x")
    ode.setfunction(1, f"-({w}**2) * (x1 - {yeq})")
    
    scheme = CrankNicolson(ode)
    T = 100
    N = 1000
    time, values = scheme.solve(T, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
    fig.suptitle(f"spring-mass system analysis ({scheme.name})\n T={T}, N={N}\n")
    scat1 = ax1.scatter(0, values[:,0][0], c="b", s=5, label=f'x=f(t)')
    line1 = ax1.plot(0, values[:,0][0])[0]
    ax1.set(xlim=[0, T], ylim=[-40, 40], xlabel="time")
    ax1.legend()
    
    scat2 = ax2.scatter(0, values[:,0][0], c="b", s=5)
    ax2.set(xlim=[-0.5, 0.5], ylim=[-40, 40])
    ax2.set_xticks([])
    
    def update1(frame):
        y1 = values[:,0][:frame]
        x = time[:frame]
        data1 = np.stack([x, y1]).T
        scat1.set_offsets(data1)
        
        line1.set_xdata(x)
        line1.set_ydata(y1)
        
        y2 = values[:,0][frame]
        x2 = time[0]
        data2 = np.stack([x2, y2]).T
        scat2.set_offsets(data2)
        return scat1, line1, scat2
    
    ani = animation.FuncAnimation(fig=fig, func=update1, frames=N, interval=3)
    
    plt.legend()
    plt.show()
    
def example11():
    "bink tank cascade"
    ode = ODE(3, 1, 0)
    ode.setinit([20, 40, 60])
    ode.setsymbols("x")
    ode.setfunction(1, "-x1/2")
    ode.setfunction(2, "x1/2 - x2/4")
    ode.setfunction(3, "x2/4 - x3/6")
    
    scheme1 = ImplicitEuler(ode)
    scheme2 = RungeKutta(ode, butcher="RK2")
    scheme3 = RungeKutta(ode, butcher="RK4")
    # scheme2 = ModifiedEuler(ode)
    # scheme3 = ImplicitEuler(ode)
    T, N = 20, 20
    time1, values1 = scheme1.solve(T, N)
    time2, values2 = scheme2.solve(T, N)
    time3, values3 = scheme3.solve(T, N)
    
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle(f"bink tank cascade T = {T}, N = {N}")
    for nn, ax in enumerate(axs.flat):
        line1, = ax.plot(time1, values1[:,nn], label=scheme1.name)
        line2, = ax.plot(time2, values2[:,nn], label=scheme2.name)
        line3, = ax.plot(time3, values3[:,nn], label=scheme3.name)
        tmp = {0: "upper", 1: "middle", 2: "lower"}
        ax.set_title(f"{tmp[nn]} tank")
        ax.set(ylim=[0, 61])
        
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    example11()
    
    