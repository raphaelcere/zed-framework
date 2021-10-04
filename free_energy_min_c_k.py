
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
	delta_g = np.zeros((mz,))
	egg = np.zeros((mz,))
	fig = np.zeros((nz,mz))
	D_ig = np.zeros((nz,mz))
	wz = np.zeros((nz,mz))
	m = np.shape(z)[0]

	#limit zero to avoid zero divisions
	c = np.double(1e-30)

	#alloc records
	d={}
	for i in range(mz):
		d["rhog{0}".format(i)]= []

	#weight
	fi = np.diag(f)[:, np.newaxis]

	# iter Z
	for i in range(Niter):
		zstep[:,:] = z

		rhog[:,] = np.dot(fi.T, z)
		for i in range(mz):
			d['rhog'+str(i)].append(rhog[i])

		for g in range(mz):
			fig[:, g] = (np.divide(np.multiply(fi, z[:,g][:, None]), rhog[g]))[:, 0]

		for g in range(mz):
			egg[g] = np.dot(np.dot(z[:,g].T, E), zstep[:,g])

		for g in range(mz):
			delta_g[g] = np.multiply(0.5, np.dot(np.dot(fig[:,g].T, D), fig[:,g]))

		for g in range(mz):
			D_ig[:,g] = np.dot(D, fig[:,g]) - delta_g[g]

		for g in range(mz):
			wz[:,g] = np.dot(W.T, z[:,g])

		for g in range(mz):
			phi_g[:,g] = np.multiply(rhog[g], np.exp(-(beta*D_ig[:, g] 
				- np.multiply(alpha*(rhog[g]**(-k)), wz[:,g]) 
				+ ((alpha*k)/2.)*np.multiply((rhog[g]**(-k-1.)),(egg[g])) 
				+ alpha*(1.-(k/2.))*(rhog[g]**(1.-k)))))

		if len(phi_g[phi_g == np.inf])!= 0:
			print('### Warning : inf value encountered in phi_g ###')
			for j in range(nz):
				phi_g[j, np.argwhere(phi_g[j,:] == np.inf)] = 1.-phi_g[j, np.argwhere(phi_g[j,:] != np.inf)].sum()

		phi_h[:,]  = phi_g[:,:].sum(axis=1)

		z[:,:] = np.divide(phi_g, phi_h[:, np.newaxis])

		if stro_mode == 'block':
			for g in range(1,mz):
				for j in rindx["gr%s" % g]:
					z[j,:] = 0.0
					z[j,g] = 1.0

		# fonctional records
		alpha_2 = alpha/2.
		k_z = np.dot(fi.T, z).dot(np.log(((z+c)/(rhog[:]+c))).T).sum()
		w_z = beta*np.dot(rhog[:], delta_g)
		c_z = alpha_2*((rhog[:]**2 - egg)/(rhog[:]**k)).sum()
		beta_z.append((w_z+c_z+k_z))

	return z, beta_z, d