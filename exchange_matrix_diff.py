import numpy as np

def toExchange(a, f, t=1):
    """
    exchange matrix
    """

    import time
    start = time.time()
    time.time()

    # Laplacian a
    #la = np.diagflat(a[:, :].sum(axis=0)[:, None]) - a[:, :]
    la = np.diag(a[:, :].sum(axis=1)) - a[:,:]
    psi_d = np.diag(np.sqrt(1./np.diag(f)))
    psi_s = np.sum(a)-np.sum(np.diag(a))
    psi = (np.dot(psi_d, la).dot(psi_d))/np.double(psi_s)

    # spectral decomposition
    v, u = np.linalg.eig(psi[:,:])
    #idx = v.argsort()[::-1]   
    #v = v[idx]
    #u = u[:,idx]

    ex_d = np.diag(np.sqrt(np.diag(f)))
    ex_e = np.diag(np.exp(-t*v))
    ex = np.dot(ex_d, u).dot(ex_e).dot(u.T).dot(ex_d)
    #ex[:,:] = np.dot(ex, ex_e)
    #ex[:,:] = np.dot(ex, u.T).dot(ex_d)

    ex[ex < 0.] = 0.

    el = ex[:,:] / ex.sum()

    #print(np.allclose(ex, ex.T, atol=1e-15))

    fl = el[:, :].sum(axis=1)    

    end = time.time()
    time = end-start
    #print("Diffusive: %f seconds" % time)

    return el, fl