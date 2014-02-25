'''
Hugh R. Wilson, "Spikes, decisions, and actions: dynamical foundations of
neuroscience", chapter 5, problem 1

With additions from Holmes 323HW14.1.pdf, problem 5.

@author: bertalan@princeton.edu
'''
import numpy as np
import matplotlib.pyplot as plt

from Integrators import integrate

def dXdt(X, t):
    R = X[0]  # len(X) = 1
    P = 20. * np.sin(np.pi * 20 * t)
    if P < 0:
        S = 0
    else:
        S = 100. * P**2 / (25 + P**2)
    return np.array([(-R + S) / .02])



fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 1, 1)

tf = 1.0

# h = 4ms

adaFig = plt.figure(figsize=(6,3))
adaAx = adaFig.add_subplot(1, 1, 1)
# X, T = integrate(dXdt, .001, 0.0, tf, (0.0,), method='ode15s')
# adaAx.plot(T, X[0,:], 'b', label='ode15s: $h=1$ [ms]')
dt = .025
X, T = integrate(dXdt, dt, 0.0, tf, (0.0,), method='ode15s')
adaAx.plot(T, X[0,:], 'r.', label='ode15s: $h=%.0f$ [ms]' % (dt * 1e3))
X, T = integrate(dXdt, dt, 0.0, tf, (0.0,), method='rungekutta')
adaAx.plot(T, X[0,:], 'g', label='RK4: $h=%.0f$ [ms]' % (dt * 1e3))
adaAx.set_title('adaptive time-stepping')

X, T = integrate(dXdt, .002, 0.0, tf, (0.0,), method='rungekutta')
ax.plot(T, X[0,:], 'k.', label='RK4: $h=2$ [ms]')
X, T = integrate(dXdt, .004, 0.0, tf, (0.0,), method='rungekutta')
ax.plot(T, X[0,:], 'r.',  label='RK4: $h=4$ [ms]')

fig2 = plt.figure(figsize=(8,4))
ax2forward  = fig2.add_subplot(1, 2, 1)
ax2backward = fig2.add_subplot(1, 2, 2)
for axis, method in zip((ax2forward, ax2backward), ('Forward Euler', 'Backward Euler')):
    axis.set_title(method)
    for h in .002, .004, .01, .02, .05:
        X, T = integrate(dXdt, h, 0.0, tf, (0.0,), method=method
                                                          .replace(' ', '')
                                                          .replace('Forward', ''))
        axis.plot(T, X[0,:], label='$h=%.3f$ [ms]' % h)
        

# Fix up the plot a bit.
showTime = 0.20
ax.set_title('Wilson 5.1. simple Naka-Rushton neuron\n'
         'Only the last %.2f [s] of integration time is shown.' % showTime)
for axis in ax, ax2forward, ax2backward, adaAx:
    d = showTime / 6
    maxy = 90
    miny = -10
    
    if axis is not ax2backward:
        axis.set_ylabel('$R(t)$')
        axis.set_yticks(np.arange(10, maxy, 10))
    else:
        axis.set_yticks([])
    axis.set_ylim(miny,maxy)
    
    axis.set_xlabel('$t$ [s]')
    axis.set_xticks(np.linspace(tf-showTime+d, tf-d, 4))
    axis.set_xlim(tf - showTime, tf)
    
ax.legend(loc='upper left')
adaAx.legend(loc='upper left')
ax2backward.legend(loc='right')
fig.subplots_adjust(bottom=.18, top=.8, left=.11)
fig2.subplots_adjust(bottom=.14, top=.9, left=.09, right=.97, wspace=.05)

fig.savefig('hw1b-wils5.1.pdf')

# The analytical solution for P=1 is
#  R(t) = \frac{50}{13}(1 - e^{-50t})

plt.show()