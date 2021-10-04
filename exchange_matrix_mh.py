import numpy as np
def toexchangemh(a, f):
    """
    exchange matrix
    """
    import time
    
    start = time.time()
    time.time()

    def ftype(e):
        return np.double(e)

    def fsum(e, ax):
        return np.sum(e[:,:], axis=ax, dtype=np.double)

    def fdiag(e):
        return np.diag(e)

    def fdiagf(e):
        return np.diagflat(e)

    def fdiv(e, q):
        return e/q

    def dmul(e, p):
        return np.dot(e, p)

    def fm(e, p):
        return e*p

    def fabs(e):
        return np.abs(e)

    def frep(e, n):
        return np.repeat(e, n)

    pis = ftype(f)
    ass = fsum(a[:, :], None)
    ais = fsum(a,0)
    fi = ftype(fdiag(f))
    n = np.shape(f)[0]

    gi = fdiv(ais, ass)
    kappa = fdiv(fi, gi)[:,None]
    v1 = frep(1, n)[:,None]
    m1 = dmul(v1, kappa.T)
    m2 = dmul(kappa, v1.T)
    mink = (fm(0.5, fabs((m1 + m2))) - 
            fm(0.5, fabs((m1 - m2))))

    laux = fdiv(fm(mink, ftype(a[:, :])), ass)
    lauxs = fsum(laux, 0)[:,None]

    emh = laux + fdiagf((fi - lauxs.T))
    #emh[emh < 0] = 0.

    fl = fdiag(fsum(emh, 0))

    end = time.time()
    time = end-start
    print("Metropolis-Hasting: %f seconds" % time)

    return emh, fl


