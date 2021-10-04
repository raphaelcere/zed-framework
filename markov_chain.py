import numpy as np

def toW(el,fl):
    """
    """
    w = np.divide(el[:,:].T,fl[:]).T
    
    return w

def w_r(w, r):
    """
    """
    o = np.zeros_like(w)
    o[:, :] = np.double(w[:, :])
    for i in range(1, r):
        o[:, :] = np.dot(np.double(o[:, :]), np.double(w[:, :]))

    return o