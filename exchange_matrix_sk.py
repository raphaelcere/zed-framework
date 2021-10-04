import numpy as np
def toexchangesk(fl, A, a, iterates):
    """
    exchange matrix Sinkhorn
    """
    Amod = np.double(A)+np.double(a)
    U = Amod
    error = []

    for step in range(int(iterates)):
        V = np.dot(np.diag(np.diag(fl)/(U).sum(axis=1)), U)
        U = np.dot(V, np.diag(np.diag(fl)/(V).sum(axis=0)))
        error_ij = np.abs((U).sum(axis=0)-(fl).sum(axis=0)).sum()+np.abs((U).sum(axis=1)-(fl).sum(axis=1)).sum()
        error.append(error_ij)
        if error_ij < 1e-100:
            break

    el = U
    return el, fl
