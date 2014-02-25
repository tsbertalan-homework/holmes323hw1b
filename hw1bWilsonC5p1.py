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


tf = 1.0


# First, a plot showing adaptive timestepping. Not that it was requested.
adapFig = plt.figure(figsize=(8.5,3.5))
adapAx = adapFig.add_subplot(1, 1, 1)
h = .025
for method, label, style in zip(
                                ('ode15s', 'rungekutta'),
                                ('ode15s', 'RK4'),
                                ('r.', 'g')): 
    X, T = integrate(dXdt, h, t0=0.0, tf=tf, X0=(0.0,), method=method)
    adapAx.plot(T, X[0,:], style, label='%s: $h=%.0f$ [ms]' % (label, h*1e3))

# Now, Runge-Kutta O(4) with various timesteps.
tstepFig = plt.figure(figsize=(8.5,3.5))
tstepAx = tstepFig.add_subplot(1, 1, 1)

hs = .001, .002, .004, .01, .02, .05
# To get something to compare against, we'll use a super-fine timestep and just
#  hope that we don't get subtractive machine-precision error.  

for h in hs:
    X, T = integrate(dXdt, h, 0.0, tf, (0.0,), method='rungekutta')
    tstepAx.plot(T, X[0,:], label='%.0f' % (h*1e3))

# Let's explicitly look at the error of the final point.
def LinInterp(T, X):
    '''
    Produce a function that will linearly interpolate X(T) to give X values
    for requested T values. Both T and X should be lists, tuples, or 1D arrays.
    '''
    def findClosest(vec, val):
        """Return the index where vec[index] is closest to val.
        >>> findClosest([2, 8, 3, 6], 5)
        3
        """
        distances = np.abs([val - x for x in vec])
        return distances.tolist().index(np.min(distances))
    def linInterp(t):
        '''A very slow interpolator.'''
        if t in T.tolist():
            return X[T.tolist().index(t)]
        else:
            i = findClosest(T, t)
            if i == 0:
                j = 1
            elif i == T.size - 1:
                j = i - 1 
            else:
                if abs(T[i+1]-t) < abs(T[i-1]-t):
                    j = i + 1
                else:
                    j = i - 1
        dxdt = (X[j] - X[i]) / (T[j] - T[i])
        return dxdt * (t - T[i]) + X[i]
    return linInterp
        
    
errFig = plt.figure(figsize=(8.5, 3.5))
errAx = errFig.add_subplot(1, 1, 1)
trueSoln, trueTimes = integrate(dXdt, .00001, 0.0, tf, (0.0,), method='rungekutta')
errs = []
hsErr = np.log(np.logspace(.001, .1, 10)) / np.log(10)
for h in hsErr:
    print h
    X, T = integrate(dXdt, h, 0.0, tf, (0.0,), method='rungekutta')
    print "making interpolator"
    linInterp = LinInterp(T, X.ravel())
    print "finding errors"
    errs.append(abs(
                    np.linalg.norm(trueSoln[0, :] - 
                                   np.array([linInterp(t) for t in trueTimes]))
                    )
                )
errAx.scatter(hsErr, errs)
errAx.set_xlabel('$log_{10}(h)$')
errAx.set_ylabel('$log_{10}(|$(true - RK4$|)$')
errAx.set_ylim(errAx.get_xlim())
errAx.set_xscale('log')
errAx.set_yscale('log')
showTime = 0.20  # We're not going to plot the whole thing.

# Now, forward/backward Euler with the same timesteps:
fbFig = plt.figure(figsize=(8.5,4))
ax2forward  = fbFig.add_subplot(1, 2, 1)
ax2backward = fbFig.add_subplot(1, 2, 2)
for axis, method in zip(
                        (ax2forward, ax2backward),
                        ('Forward Euler', 'Backward Euler')
                        ):
    axis.set_title(method)
    for h in hs:
        X, T = integrate(dXdt, h, 0.0, tf, (0.0,), method=method
                                                          .replace(' ', '')
                                                          .replace('Forward', ''))
        axis.plot(T, X[0,:], label='%.0f' % (h*1e3))


# Fix up the plots a bit.
for axis in tstepAx, ax2forward, ax2backward, adapAx:
    d = showTime / 6
    maxy = 90
    miny = -10
    
    if axis is not ax2backward:  # We'll do something different with this.
        axis.set_ylabel('$R(t)$')
        axis.set_yticks(np.arange(10, maxy, 10))
    else:
        axis.set_yticks([])
    axis.set_ylim(miny,maxy)
    
    axis.set_xlabel('$t$ [s]')
    axis.set_xticks(np.linspace(tf-showTime+d, tf-d, 4))
    axis.set_xlim(tf - showTime, tf)
tstepAx.legend(loc='upper left')
adapAx.legend(loc='upper left')
ax2backward.legend(loc='right')
fbFig.subplots_adjust(bottom=.14, top=.9, left=.1, right=.99, wspace=.05)
tstepFig.subplots_adjust(bottom=.18, top=.99, left=.08, right=.99)
adapFig.subplots_adjust(bottom=.18, top=.99, left=.08, right=.99)

# Save them.
tstepFig.savefig('hw1b-wils5_1.pdf')
fbFig.savefig('hw1b-5-forwardBackward.pdf')
adapFig.savefig('hw1b-5-adaptive.pdf')
errFig.savefig('hw1b-5-error.pdf')

# The analytical solution for P=1 is
#  R(t) = \frac{50}{13}(1 - e^{-50t})

plt.show()