'''
Created on Feb 24, 2014, 1:04:36 PM

@author: bertalan
'''

import numpy as np  # Gives us matrices and linear algebra.
import logging
logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.INFO, # Turn on/off debug by switching INFO/ERROR.
                    name='log')


def integrate(dXdt, h, t0, tf, X0, method='euler', newtontol=1e-6):
    """
    Integrate the differential equation dXdt from t0 to tf in steps of size h,
    and initial condition X0. method can be one of 'euler', 'rungeKutta', or
    'backwardEuler'; not case-sensitive.
    
    If 'backwardEuler' is used, newtontol specifies the tolerance for the
    implicit solver.
    
    Returns the history of states and the times at which they occured.
    
    dXdt must accept a state and a time. E.g, this linear ODE:
    
    >>> def dXdt(X, t):
    ...     A = np.array([[1, -2],
    ...                   [3, -4]])  # Eigenvalues are -2 and -1, so the origin is stable.
    ...     X = np.array(X).reshape((2,1))
    ...     return np.dot(A, X)
    >>> h=.1; t0=0; tf=4; X0=(4,8)
    >>> X, T = integrate(dXdt, h, t0, tf, X0)
    >>> print X.shape, T.shape
    (2, 41) (41,)
    >>> assert T[0] == t0
    >>> assert T[-1] == tf
    >>> assert (X[:,0] == X0).all  # ".all" checks that all boolean-valued
    ...                            #  array elements are True
    >>> from matplotlib import pyplot as plt
    >>> f = plt.figure()
    >>> a = f.add_subplot(1, 1, 1)
    
    Do a phase portrait.
    >>> p = a.plot(X[0, :], X[1, :])
    
    Mark the initial condition.
    >>> s = a.scatter(X0[0], X0[1])
    
    Try the ODE45 "equivalent"
    >>> X, T = integrate(dXdt, h, t0, tf, X0, method='ode45')
    >>> assert X.size > 2
    
    """
    
    # Reshape the initial condition as a column vector. Allows us to accept
    #  row-vector numpy arrays, lists, tuples, or other list-like things.
    #  This might fail if X0 is something stupid, like a list-of-lists,
    #  or a string.
    X0 = np.array(X0)
    N = X0.size
    X0 = X0.reshape((N,1))
    
    # Initialize the history arrays.
    #  for more general, adaptive methods, we couldn't do this. At least, not
    #  all at once. The +h allows us to include both t0 and tf in the history.
    T = np.arange(t0, tf+h, h)
    X = np.empty((N, T.size))
    X[:,0] = X0.ravel()
    
    f = lambda tn, xn: dXdt(xn, tn)  # symbols chosen to match Wikipedia's RK4
    h = h
    
    # some method-codes
    ode45 = 'dopri5' 'ode45'
    ode15s = 'vode' 'ode15s'
    
    if method.lower() == 'rungekutta':
        for n in xrange(T.size-1):
            xn = X[:, n].reshape((N,1))
            tn = T[n]
            k1 = f(tn, xn)
            k2 = f(tn + h/2., xn + h * k1 / 2.)
            k3 = f(tn + h/2., xn + h * k2 / 2.)
            k4 = f(tn + h,    xn + h * k3)
            X[:, n+1] = (xn + 1/6. * h * (k1 + 2. * k2 + 2. * k3 + k4)).ravel()
            
    elif method.lower() == 'backwardeuler':
        for k in xrange(T.size-1): # We're using k because Wikipedia does.
            xk = X[:, k].reshape((N,1))
            tk = T[k]
            tk1 = tk + h
            xk10 = xk  # initial iterate
            # While both Numpy and Scipy have fancier things available, we'll
            #  deliberately use Newton-Rhapson, without putting forth the effort
            #  to write our own.
            from scipy.optimize import newton
            def zeroMe(xk1):
                return xk + h * f(tk1, xk1) - xk1
            xk1 = newton(zeroMe, xk10, tol=newtontol)#, maxiter=...)
            X[:, k+1] = xk1.ravel()
            
    elif method.lower() in ode45 + ode15s:
        # Awkward to use. scipy.integrate.odeint is easier. This code is adapted
        #  from the [SciPy-User] mailing list:
        #  http://mail.scipy.org/pipermail/scipy-user/2011-March/028683.html
        # It's safe to use a quite large h value for the ode15s method, since
        #  the real step size is chosen automatically by SciPy. This can be
        #  verified by using this method on the Naka-Rushton problem with a
        #  step size larger than, say 1/6 the oscillatory period, and
        #  scatter-plotting R vs t. The parts with large |dR/h|
        #  automatically get smaller timesteps.
        from scipy.integrate import ode
        solver = ode(f)
        if method.lower() in ode45:
            solver.set_integrator('dopri5', nsteps=1)
            from warnings import filterwarnings
            filterwarnings("ignore", category=UserWarning)
        if method.lower() in ode15s:
            solver.set_integrator('vode', method='bdf', order=15)
        solver.set_initial_value(X0, t0)
        solver._integrator.iwork[2] = -1
        T = [t0]
        X = [X0]
        while solver.t < tf:
            solver.integrate(tf, step=True)  # We should get *at least*
            T.append(solver.t)  #  as many steps as the other methods; perhaps
            X.append(solver.y.reshape((N,)))  #  more.
        T = np.array(T)
        X = np.array(X).T
    
    else:  # assume forward Euler if not ^^. Later, we might add, say,
        #     Adams-Bashforth. Or perhaps backwards Euler. 
        for n in xrange(T.size-1):
            xn = X[:, n].reshape((N,1))
            tn = T[n]
            X[:, n+1] = (xn + f(tn, xn) * h).ravel()
    
    return X, T


if __name__ == '__main__':
    '''
    The code in this block only runs if this file is invoked as a script;
    not if things from this file are imported elsewhere. Guido van Rossum
    disapproves of this usage, but it's just so handy.
    '''
    # Test that documentation examples are correct.
    import doctest
    doctest.testmod()
    
    # Do a pseudo-triangle-wave thing.
    def dXdt(X, t):
        X = np.array(X).reshape((2,1))
        x = X[0,0]
        y = X[1,0]
        return (np.array([np.cos(.05423*t), np.sin(t)]) +\
                np.array([np.cos(1.121235432*x), np.sin(.123*y)])).reshape((2,1))
    X, T = integrate(dXdt, h=.1, t0=0, tf=4, X0=(42, 68), method='rungekutta')
    from matplotlib import pyplot as plt
#     f = plt.figure()
#     a = f.add_subplot(1, 1, 1)
#     a.plot(X[0,:], X[1,:])  # will plot both X[0,:], and X[1,:] vs. T

    # Display any plots that may have been generated.
#     plt.show()
