from univariate_rng import UnivariateRNG
import numpy as np 
import matplotlib.pyplot as plt


class LogisticRegression(object): 
	def __init__(self, n, mx1, vx1, mx2, vx2, my1, vy1, my2, vy2, basis=3, learning_rate=1e-2): 
		self.n = n
		self.rng = UnivariateRNG()
		self.basis = basis 
		self.params = basis * 2
		self.lr = learning_rate

		rand_points = lambda nb, m, v : \
			np.array([self.rng.rand_normal(m, v) for i in range(nb)])
		
		self.x1 = rand_points(n, mx1, vx1)
		self.y1 = rand_points(n, my1, vy1)

		self.x2 = rand_points(n, mx2, vx2)
		self.y2 = rand_points(n, my2, vy2)

		self.phi = self.make_poly_design_matrix((self.x1, self.y1), (self.x2, self.y2))
		self.w = np.array([self.rng.rand_normal(0., 0.1) for i in range(self.params)])
		self.d = np.array([0. for i in range(n)] + [1. for i in range(n)])
		shuf = np.append(self.phi, np.vstack(self.d), axis=1)
		np.random.shuffle(shuf)
		self.phi = shuf[:, :self.params]
		self.y = shuf[:, self.params]

	def make_poly_design_matrix(self, xy1, xy2):
		phi = np.zeros((2 * self.n, self.params))

		for ti in range(2):
			v = xy1[ti] 
			for i in range(self.n): 
				for j in range(self.basis): 
					phi[i, ti * self.basis + j] = v[i]**j
		for ti in range(2):
			v = xy2[ti] 
			for i in range(self.n): 
				for j in range(self.basis): 
					phi[i + self.n, ti * self.basis + j] = v[i]**j
		return phi 

	def logistic(self, x): 
		return 1 / (1 + np.exp(-x))

	def __call__(self, x, y): 
		return 1/(1+np.exp(- np.dot(np.array([x**i for i in range(self.basis)] + [y**i for i in range(self.basis)]), self.w)))

	def compute_gradients(self): 
		grad = np.zeros(self.w.shape)
		for j in range(self.params): 
			for i in range(self.n): 
				grad[j] += self.phi[i, j] * \
					(self.logistic(np.dot(self.phi[i, :], self.w)) - self.y[i]) 
		return grad / self.n

	def compute_hessian(self): 
		hess = np.zeros((self.params, self.params))
		for j in range(self.params): 
			for k in range(self.params):
				for i in range(self.n):
					# hess[j, k] += self.phi[i, j]*self.phi[i, k] *\
					# 	(1 / (1 + np.exp(-np.dot(self.phi[i,:], self.w)))**2 - self.y[i]) 
					hess[j, k] += self.phi[i, j] * self.phi[i, k] * (self.logistic(np.dot(self.phi[i, :], self.w)) * (1 - self.logistic(np.dot(self.phi[i, :], self.w))))
		return hess

	def newton_descent(self, grad_w, hess_w): 
		# Matrix suppose to be invertible here
		self.w -= self.lr * np.dot(np.linalg.inv(hess_w), grad_w) 

	def gradient_descent(self, grad_w): 
		self.w -= self.lr * grad_w 

	def optimize_once(self): 
		grad_w = self.compute_gradients() 
		hess_w = self.compute_hessian() 
		# # print(hess_w)
		# self.gradient_descent(grad_w)
		if np.abs(np.linalg.det(hess_w))> 1e-5: 
			self.newton_descent(grad_w, hess_w) 
		# print(np.mean(grad_w))
			print("used newton_descent")
			return np.linalg.norm(hess_w)
		else: 
			# print(grad_w)self
			self.gradient_descent(grad_w)
			return np.linalg.norm(grad_w)


def confusion_matrix(predicted, expected, n_classes=None): 
	if n_classes is None: 
		n_classes = int(np.amax([np.amax(expected), np.amax(predicted)]) - np.amin([np.amin(expected), np.amin(predicted)])) + 1
	m = np.zeros((n_classes, n_classes))
	for i in range(predicted.shape[0]):
		m[int(predicted[i])][int(expected[i])] += 1.
	return m / (predicted.shape[0] / 2)

if __name__ == '__main__': 
	n = 150
	res = 75

	c_C1 = (1,1)
	v_C1 = (.3, .7)
	c_C2 = (-1,-1)
	v_C2 = (.5, .5)

	l = LogisticRegression(n,c_C1[0],v_C1[0],c_C2[0],v_C2[0],c_C1[1],v_C1[1],c_C2[1],v_C2[1])
	ran = np.linspace(0,3,n)

	x_min = np.min([np.min(l.x1), np.min(l.x2)])
	x_max = np.max([np.max(l.x1), np.max(l.x2)])
	y_min = np.min([np.min(l.y1), np.min(l.y2)])
	y_max = np.max([np.max(l.y1), np.max(l.y2)])

	x = np.linspace(x_min, x_max, res)
	y = np.linspace(y_min, y_max, res)
	X, Y = np.meshgrid(x, y)

	for i in range(1000000): 
		g = l.optimize_once()

		if i % 10 == 0: 
			pred = 0
			for i in range(n): 
				pred += l(l.x1[i], l.y1[i]) > 0.5
			for i in range(n): 
				pred += l(l.x2[i], l.y2[i]) < 0.5
		

		if i % 100 == 0: 
			print(l.w)
		if i % 50 == 1: 
			plt.close()

			pred = 0
			for i in range(n): 
				pred += np.abs(l(l.x1[i], l.y1[i]) - l.d[i]) < 0.5
			for i in range(n): 
				pred += np.abs(l(l.x2[i], l.y2[i]) - l.d[i + n]) < 0.5
			print("acc : %f %%"%(100. * np.mean(pred) / (2.*n)))

			p = np.array([int(l(l.x1[i], l.y1[i]) > 0.5) for i in range(n)] + [int(l(l.x2[i], l.y2[i]) > 0.5) for i in range(n)])
			e = l.d

			cm_mat = confusion_matrix(p , e)
			print(cm_mat)
			
			plt.subplot(2,1,1)
			plt.imshow(cm_mat, cmap=plt.cm.gray_r)
			plt.colorbar()
			plt.subplot(2,1,2)
			plt.plot(l.x1, l.y1, 'o')
			plt.plot(l.x2, l.y2, 'x')

			Z = np.zeros((res, res))
			for i in range(res): 
				for j in range(res): 
					Z[i, j] = l(x[i], y[j]) 


			c = plt.contourf(X,Y,Z,cmap=plt.cm.BuPu)
			cbar = plt.colorbar(c)

			print("min : %f, max : %f"%(np.min(Z), np.max(Z)))
			# plt.plot(ran, [l(x) for x in ran])
			plt.show(False)
		
		if g < 1e-2: 
			break