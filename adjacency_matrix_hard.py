import numpy as np

def diagonals(shape_x, shape_y, n):
	neigh = np.repeat([1.],n)
	neigh_a = np.repeat([1.], n)
	for i in range(shape_x-1, len(neigh_a), shape_x):
		neigh_a[i] = 0.

	neigh_b = np.repeat([1.], n)
	for i in range(0, len(neigh_b), shape_x):
		neigh_b[i] = 0.

	neigh_b[0] = 0.

	return neigh, neigh_a, neigh_b

def replace_diagonals(a, neigh, neigh_a, neigh_b, shape_x, shape_y, n):
	n = shape_x*shape_y
	inds = [1,shape_x+1,-1,-(shape_x+1)]
	rows, cols = np.indices((n,n))

	for i in inds:
		a += np.diagflat(neigh_a, i)[:n,:n]

	inds = [shape_x-1,-(shape_x-1)]
	for i in inds:
		a += np.diagflat(neigh_b, i)[:n,:n]

	inds = [shape_x,-(shape_x)]
	for i in inds:
		a += np.diagflat(neigh, i)[:n,:n]

	return a
