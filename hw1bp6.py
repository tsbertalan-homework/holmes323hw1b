'''
Created on Feb 24, 2014, 3:58:19 PM

@author: bertalan
'''

import numpy as np
import matplotlib.pyplot as plt
from Integrators import integrate, logging

def FitzhughNagumo(Iapp):
    '''This closure gives the RHS function for a particular applied current value.'''
    def dXdt(X, t):
        '''t is not used (the ODE is autonomous)'''
        v = X[0]
        r = X[1]
        tv = .1
        tr = 1
        dvdt = (v - v**3 / 3 - r + Iapp) / tv
        drdt = (-r + 1.25 * v + 1.5) / tr
        return np.array([dvdt, drdt]) 
    return dXdt


fig = plt.figure()

Iapps = 0, 0.9, 1.01, 1.5, 2.05, 2.1
# for Iapp = 0, the fixed point is (-3/2, -3/8)
axes = []
for i in range(6):
    axes.append(fig.add_subplot(2, 3, i+1))

for ax, Iapp in zip(axes, Iapps):
    logging.info('Iapp=%f' % Iapp)
    dXdt = FitzhughNagumo(Iapp)
    ax.axhline(0, color='red')
    ax.axvline(0, color='red')
    ax.scatter(-3./2, -3./8, color='blue')
    for v0 in np.linspace(-2, 2.5, 8):
#         logging.info('v0=%s' % str(v0))
        for r0 in np.linspace(-1, 3.5, 8):
            X0 = v0, r0
            logging.info('X0=%s' % str(X0))
            # Uncomment these two lines to verify the Euler solution with RK4:
            #X, T = integrate(dXdt, .01, 0, 10.0, X0, method='rungekutta')
            #ax.plot(X[0,:], X[1,:], 'r.')
            X, T = integrate(dXdt, .01, 0, 10.0, X0, method='euler')
            ax.plot(X[0,:], X[1,:], 'k')
            ax.scatter(X[0,0], X[1,0], color='green')  # the initial condition...
            ax.scatter(X[0,-1], X[1,-1], color='red')  # ...and the final point
            ax.set_title('$I_\mathrm{app} = %.2f$' % Iapp)
for i in 0, 1, 2:
    axes[i].set_xticks([])
for i in 1, 2, 4, 5:
    axes[i].set_yticks([])
for i in 0, 3:
    axes[i].set_ylabel('$r(t)$')
for i in 3, 4, 5:
    axes[i].set_xlabel('$v(t)$')
for ax in axes:
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 4)

ax.legend()

fig.savefig('hw1bp6-flows.pdf')

plt.show()
