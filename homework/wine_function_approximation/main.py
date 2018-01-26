import numpy as np
from scipy.linalg import solve
from matplotlib import pylab as pylab
from scipy import interpolate


def f(x): return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def row(x): return np.array([1., x, x ** 2, x ** 3])


x_orig = np.arange(1, 15, 0.5)
y_orig = f(x_orig)

x_approx = np.array([1., 4., 10., 15.])
a = np.array([row(x_approx[0]), row(x_approx[1]), row(x_approx[2]), row(x_approx[3])])
b = np.array(f(x_approx))
y_approx = solve(a, b)

result_file = open('result.txt', 'w')
result_file.write(np.array_str(y_approx))
result_file.close()

pylab.figure()
pylab.plot(x_orig, y_orig, 'r', label='original')
pylab.plot(x_approx, y_approx, 'bo', label='approximation')
inter = interpolate.interp1d(x_approx, y_approx, kind='quadratic')
x_approx_quad = np.arange(0, 15, 0.1)
y_approx_quad = f(x_approx_quad)
pylab.plot(x_approx_quad, y_approx_quad, 'g-.', label='qudaratic')

pylab.legend()
pylab.show()
