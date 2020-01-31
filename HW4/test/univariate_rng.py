import numpy as np
import time

class UnivariateRNG(object): 
	''' 
	The functions __init__ and rand_uniform 
	refers to http://paulbourke.net/miscellaneous/random/, 
	and are based on the FORTRAN algorithms from 
	George Marsaglia and Arif Zaman, Florida State University. 
	The versions have been slightly modified to work in 
	Python. 
	''' 

	def __init__(self, ij=None, kl=None): 
		if ij is None or kl is None: 
			ij = int(time.time() * np.exp(-time.clock()))
			kl = int(-np.log(time.clock() / (0.001 * time.time())))

		if ij >= 31328: 
			ij = ij % 31328 + 2 
		elif ij < 0: 
			ij = (-ij % 31328) + 2 
		
		if kl >= 30081: 
			kl = kl % 30081 + 2 
		elif kl < 0: 
			kl = (-kl % 30081) + 2 

		i = (ij / 177) % 177 + 2
		j = (ij % 177) + 2
		k = (kl / 169) % 178 + 1
		l = (kl % 169)

		self.u = np.zeros(97);
		for ii in range(97): 
			s = 0.0
			t = 0.5
			for jj in range(24): 
				m = (((i * j) % 179) * k) % 179
				i = j
				j = k
				k = m
				l = (53 * l + 1) % 169
				if (l * m % 64) >= 32:
					s += t
				t *= 0.5
			self.u[ii] = s
		self.c    = 362436.0 / 16777216.0
		self.cd   = 7654321.0 / 16777216.0
		self.cm   = 16777213.0 / 16777216.0
		self.i97  = 97
		self.j97  = 33

	def rand_uniform(self, a=0, b=1): 
		uni = self.u[self.i97 - 1] - self.u[self.j97 - 1]
		if( uni <= 0.0 ):
			uni += 1
		self.u[self.i97 - 1] = uni
		self.i97 -= 1
		if self.i97 == 0: 
			self.i97 = 97
		self.j97 -= 1
		if self.j97 == 0:
			self.j97 = 97
		self.c -= self.cd
		if self.c < 0.0:
			self.c += self.cm
		uni -= self.c
		if uni < 0.0:
			uni += 1

		if a != 0 or b != 1: 
			uni = uni * (b - a) + a
		return uni


	def rand_normal(self, mean, var): 
		''' 
		rand_normal() uses the Marsaglia polar method to generate a standard normally distributed
		outcome, and then scale it using mean and var in argument to follow a N(mean, var) 
		distribution
		''' 
		assert(var >= 0)
		S = 0 
		X = 0
		while S >= 1 or S == 0: 
			U = self.rand_uniform(-1,1)
			S = U ** 2 + (self.rand_uniform(-1,1))**2 
			if S != 0:
				temp =  - 2 * np.log(S) / S
				if temp >= 0: 
					X = U * np.sqrt(temp)

		return float(mean) + X * float(np.sqrt(var))

if __name__ == "__main__": 
	r = UnivariateRNG() 



	g = np.array([r.rand_normal(0,1) for i in range(1000)])
	print(np.mean(g))
	print(np.var(g))