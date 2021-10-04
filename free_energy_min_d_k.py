import numpy as np

def mink(k, stro_mode, z0, f, D, W, E, Niter, alpha, beta, rindx):
	beta_z = []
	z = z0[:,:]
	mz = np.shape(z)[1]
	nz = np.shape(z)[0]
	phi_g = np.zeros((nz,mz))
	phi_h = np.zeros((nz,))
	zstep = np.zeros((nz,mz))
	rhog = np.zeros((mz,))
	rhogstep = np.zeros((mz,))
	delta_g = np.zeros((mz,))
	fig = np.zeros((nz,mz))
	D_ig = np.zeros((nz,mz))
	lap = np.zeros((nz,mz))
	eps = np.zeros((nz,mz))
	deltaeps = np.zeros((nz,mz))
	m = np.shape(z)[0]
	tmp = np.inf

	#limit zero to avoid zero divisions
	c = np.double(1e-100)

	#alloc records
	d={}
	for i in range(mz):
		d["rhog{0}".format(i)]= []

	#weight
	fi = np.diag(f)[:, np.newaxis]

	# iter Z
	for step in range(Niter):

		mg = np.shape(z)[1]
		if step == 0:
			rhogstep[:,] = np.dot(fi.T, z)
			zstep[:,:] = z[:,:]
		else:
			rhogstep[:,] = np.dot(fi.T, zstep)

		for g in range(mz):
			fig[:, g] = (np.divide(np.multiply(fi, z[:,g][:, None]), rhogstep[g]))[:, 0]

		for g in range(mz):
			delta_g[g] = np.multiply(0.5, np.dot(np.dot(fig[:,g].T, D[:,:]), fig[:,g]))

		for g in range(mz):
			D_ig[:,g] = np.dot(D[:,:], fig[:,g]) - delta_g[g]

		for g in range(mz):
			lap[:,g] = z[:,g] - np.dot(W.T, z[:,g])

		for g in range(mz):
			eps[:,g] = np.multiply(0.5, np.dot(E.T, (z[:,g] - zstep[:,g])**2.))

		for g in range(mz):
			phi_g[:,g] = np.multiply(rhogstep[g], np.exp(-(np.multiply(beta, D_ig[:, g]) 
				+ np.multiply(np.multiply(alpha,(rhogstep[g]**-k)), lap[:, g]) 
				- np.multiply(((alpha*k)/2.), np.multiply((rhogstep[g]**(-k-1)),(eps[:, g]))))))

		phi_h[:,]  = phi_g[:,:].sum(axis=1)

		z[:,:] = np.abs(np.divide(phi_g, phi_h[:, np.newaxis]))

		#print z.sum(axis=1).sum() == 309.0

		if stro_mode == 'block':
			z[rindx,:] = z0[rindx,:]
			# for g in range(1,mz):
			# 	for j in rindx["gr%s" % step]:
			# 		z[j,:] = 0.0
			# 		z[j,i] = 1.0

		# fonctional records
		alpha_2 = alpha/2.

		#k_z = np.dot(fi.T, z).dot(np.log(((z+c)/(rhogstep[:]))).T).sum()
		k_z = np.multiply(rhogstep[:], np.log(np.divide(z[:,:], rhogstep[:]))).sum()
		if k_z == -np.inf:
			print('K[Z] is'+str(k_z)+' Numpy log handles the floating-point negative zero as an infinitesimal negative number, conforming to the C99 standard')
			break

		#w_z = beta*np.dot(rhogstep, delta_g)
		w_z = np.multiply(rhogstep, delta_g).sum()

		#g_z = alpha_2*np.dot(E, ((z-zstep)**2)).sum()*0.5
		for g in range(mz):
			deltaeps[:,g] = np.multiply(0.5, np.dot(E.T, (z[:,g] - zstep[:,g])**2.))

		g_z = np.divide(deltaeps[:,:],rhogstep[:]).sum()
		mf = beta*w_z+alpha_2*g_z+k_z
		beta_z.append((mf))
		zstep[:,:] = z[:,:]

	return z, beta_z, d