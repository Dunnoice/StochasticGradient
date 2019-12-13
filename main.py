import numpy as np
import random

import stochastic_gradient as SG

np.random.seed(0)


class SGL(SG.StochasticGradient):
	def __init__(self, sample, learning_rate, forgetting_rate, weights_precision, quality_precision):
		super().__init__(sample, learning_rate, forgetting_rate, weights_precision, quality_precision)
		self.w_log, self.q_log, self.p_log = [], [], []

	def _calc_step(self):
		self.p_log.append(super()._calc_step())
		self.w_log.append(list(self.weights))
		self.q_log.append(self.quality)

	def _get_precedent_pos(self):
		return random.randrange(len(self.sample))

	def loss(self, y1, y2=1):
		"""
		Loss function

		error

		bicubic

		:return: (y1 - y2) ** 2
		"""
		return (y1 - y2) ** 2

	def loss_diff(self, y1, y2=1):
		"""
		loss derivative by y1

		:return: 2 * (y1 - y2)
		"""
		return 2 * (y1 - y2)

	def info(self):
		print('quality:', self.quality)
		print('weights:', self.weights)
		print('errors:', list(self.errors))
		q_log = self.q_log.copy()
		q_log.reverse()
		print('q.logr:', q_log)
		w_log = self.w_log.copy()
		w_log.reverse()
		print('w.logr:', w_log)


class SGvect(SGL):
	"""
	Stochastic Gradient

	uses vector multiplication
	"""

	def algorithm(self, weights, x):
		"""
		:return: <w, x>
		:rtype: float
		"""
		result = np.dot(weights, x)
		return round(result)

	def _gradient_descent(self, index):
		# {'}loss * x[i] * y[i]
		alg = self._algorithm(index)
		loss = self.loss_diff(alg, self.sample[index].y)
		xy = [x_i * self.sample[index].y for x_i in self.sample[index].x]
		lxy = [loss * xy_i for xy_i in xy]
		return lxy


class SGdiff(SGL):
	"""
	Stochastic Gradient

	weights[0] is reserved for decision threshold

	x should start from 1 in order to multiply to weights
	x[0] = artificial constant attribute
	"""

	def activate(self, z):
		"""
		Activation function

		phi

		scalar from z

		:param z: list[float] | float
		:return: z
		"""
		result = 0
		if type(z) is list:
			for z_i in z:
				result += z_i
		else:
			result = z
		return result

	def activate_diff(self, z):
		"""
		activation derivative

		:return: 1
		"""
		return 1

	def algorithm(self, weights, x):
		"""
		Applies activation function to x and weights

		:return: activate(sum(w[j] * x[j] - w[0], 1, len(x)))
		:rtype: float
		"""
		result = 0
		for j in range(len(x)):
			result += weights[j] * x[j] - weights[0]
		result = self.activate(result)
		return round(result)

	def _gradient_descent(self, index):
		# diff
		# {'a}loss * {'}activate(<w, x[i]>) * x[i]
		alg = self._algorithm(index)
		loss = self.loss_diff(alg, self.sample[index].y)
		wx_vm = np.dot(self.weights, self.sample[index].x)
		act = self.activate_diff(wx_vm)
		la = loss * act
		lax = [la * x_i for x_i in self.sample[index].x]
		return lax


file1 = 'samples/Wine Quality Data Set/winequality-red.csv'

dataset1 = np.loadtxt(file1, delimiter=';', skiprows=1)
names1 = np.genfromtxt(file1, delimiter=';', dtype=str, max_rows=1)

rate_learn = 1e-4
rate_forget = SG.forgetting_rate(dataset1)
precision_w = 5e-2
precision_q = 3

options1 = {
	'sample': SG.Sample(dataset1),
	'learning_rate': rate_learn,
	'forgetting_rate': rate_forget,
	'weights_precision': precision_w,
	'quality_precision': precision_q
}

options1_diff = dict(options1)
options1_diff['sample'] = SG.Sample(dataset1, add_const_attr=True)

sg1_diff = SGdiff(**options1_diff)
print('Calculate:', sg1_diff.calculate())
sg1_diff.info()

# sg1_vect = SGvect(**options1)
# print('Calculate:', sg1_vect.calculate())
# sg1_vect.info()

import matplotlib.pyplot as plt

x = np.array([precedent.x for precedent in options1['sample']])
y = np.array([precedent.y for precedent in options1['sample']])

sg2y = np.array([sg1_diff.algorithm(sg1_diff.weights, precedent.x) for precedent in options1['sample']])

# plt.plot(y, 'r*', sg2y, 'b.', alpha=0.1)
plt.plot(sg2y, 'b.', alpha=0.1)

# plt.show()
