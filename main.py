import numpy as np
import random

import stochastic_gradient as SG
import loggers as lgs

np.random.seed(0)


class SGvect(SG.StochasticGradient):
	"""
	Stochastic Gradient

	uses vector multiplication
	"""

	def __init__(self, sample, learning_rate, forgetting_rate, weights_precision, quality_precision):
		"""
		:param Sample sample: [[x ... , y]]
		:param forgetting_rate: functional smoothing
		"""
		super().__init__(sample, learning_rate, forgetting_rate, weights_precision, quality_precision)

		self.w_lgr = lgs.ValueL(self.weights, np.array, empty_value=[])
		self.q_lgr = lgs.ValueL(self.quality, float, empty_value=0)
		self.e_lgr = lgs.ValueL(self.errors, list, empty_value=[])

	def _get_precedent_pos(self):
		return random.randrange(len(self.sample))

	def algorithm(self, weights, x):
		"""
		:return: <w, x>
		:rtype: float
		"""
		result = np.dot(weights, x)
		return round(result)

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

	def _gradient_descent(self, index):
		# {'}loss * x[i] * y[i]
		alg = self._algorithm(index)
		loss = self.loss_diff(alg, self.sample[index].y)
		xy = [x_i * self.sample[index].y for x_i in self.sample[index].x]
		lxy = [loss * xy_i for xy_i in xy]
		return lxy

	def calculate(self):
		pass


class SGdiff(SGvect):
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
		vectm = np.dot(self.weights, self.sample[index].x)
		act = self.activate_diff(vectm)
		la = loss * act
		lax = [la * x_i for x_i in self.sample[index].x]
		return lax


def info(sg):
	print('quality:', sg.quality)
	print('weights:', sg.weights)
	print('q.logr:', sg.q_lgr.logr)
	print('w.logr:', sg.w_lgr.logr)


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

sg1 = SGvect(**options1)
print('Calculate:', sg1.calculate())
info(sg1)

options2 = dict(options1)
options2['sample'] = SG.Sample(dataset1, add_const_attr=True)

sg2 = SGdiff(**options2)
print('Calculate:', sg2.calculate())
info(sg2)


import matplotlib.pyplot as plt

x = np.array([precedent.x for precedent in options1['sample']])
y = np.array([precedent.y for precedent in options1['sample']])

sg2y = np.array([sg2.algorithm(sg2.weights, precedent.x) for precedent in options1['sample']])

# plt.plot(y, 'r*', sg2y, 'b.', alpha=0.1)
plt.plot(sg2y, 'b.', alpha=0.1)

# plt.show()
