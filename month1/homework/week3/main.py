import math
import numpy as np
from scipy.linalg import solve
from matplotlib import pylab as pylab
from scipy import interpolate
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


def f(x): return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def h(x): return np.rint(f(x))


def display_function(func):
    x = np.arange(1, 30.1, 0.1)
    y = func(x)

    pylab.figure()
    pylab.plot(x, y)
    pylab.show()


x0 = np.array([2.0])
result_f_bfgs_2 = minimize(f, x0, method='BFGS')
# print round(result_f_bfgs_2.fun, 2)
# 1.75

x0 = np.array([30.0])
result_f_bfgs_30 = minimize(f, x0, method='BFGS')
# print round(result_f_bfgs_30.fun, 2)
# -11.9

# 1: 1.75 -11.9

bounds = [(1.0, 30.0)]
result_f_de = differential_evolution(f, bounds)
# print result_f_de
# 2: -11.9

x0 = np.array([30.0])
result_h_bfgs = minimize(h, x0, method='BFGS')
# print result_h_bfgs
# -6.


bounds = [(1.0, 30.0)]
result_h_de = differential_evolution(h, bounds)
# print result_h_de
# -12.
# 3: -6. -12.

# display_function(f)
# display_function(h)
