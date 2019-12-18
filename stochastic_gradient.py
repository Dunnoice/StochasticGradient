import numpy as np

import collections


class Precedent(collections.namedtuple('Precedent', 'x y')):
	@property
	def value(self):
		return np.array([self.x, self.y])


class Sample(tuple):
	"""
	Formatted sample
	"""

	def __new__(cls, dataset, *, y_pos=None, x_end=None, add_const_attr=False):
		"""
		:type dataset: list[float] | numpy.ndarray[float]
		:param add_const_attr: x[0] = artificial constant attribute (default = 1), x starts from 1 in order to multiply to weights
		:param y_pos: position of single output (default = last in row); for other outputs create new sample
		:param x_end: slicing index (last x + 1) of input (default = y_pos)
		"""
		if y_pos is None:
			y_pos = len(dataset[0]) - 1
		if x_end is None:
			x_end = y_pos
		values = []
		for sample_i in dataset:
			x_i = sample_i[0:x_end]
			if add_const_attr:
				x_i = np.append(1., x_i)
			values.append(Precedent(x_i, sample_i[y_pos]))
		return super().__new__(cls, values)


class Base:
	sample: Sample
	rate_learning: float
	rate_forgetting: float
	weights: [float]
	quality: float
	errors: [float]
	precision_weights: float
	precision_quality: float

	def __init__(self, sample, learning_rate, forgetting_rate, quality_precision, weights_precision=0.0):
		"""
		:param Sample sample: [[x ... , y]]
		:param learning_rate: affects weights
		:param forgetting_rate: affects quality; Q smoothing
		"""
		self.sample = sample
		self.rate_learning = learning_rate
		self.rate_forgetting = forgetting_rate
		self.precision_weights = weights_precision
		self.precision_quality = quality_precision

		self.iteration = 0
		self.weights = self._init_weights()
		self._prev_weights = 0
		self.errors = self._init_errors()
		self.quality = self.q()
		self._prev_quality = 0
		self._prev_quality_diff = 0

	def _init_errors(self):
		length = len(self.sample)
		errors = np.zeros(length)
		for i in range(length):
			errors[i] = self._loss(i)
		return errors

	def q(self):
		"""
		Empirical risk

		assessment of functional

		loss of sample

		:rtype: float
		"""
		return np.sum(self.errors)

	def _init_weights(self):
		"""
		:rtype: list[float]
		"""
		precedent_length = len(self.sample[0].x)
		shape = precedent_length

		result = np.zeros(shape)
		return result

	def _set_weights(self, new_weights):
		self._prev_weights = self.weights
		self.weights = new_weights

	def _set_quality(self, new_quality):
		self._prev_quality = self.quality
		self.quality = new_quality

	def _algorithm(self, index):
		"""
		Calculates algorithm for precedent on index

		:return: algorithm(w, x[i])
		"""
		return self.algorithm(self.weights, self.sample[index].x)

	def _loss(self, index):
		"""
		Calculates loss of algorithm for precedent on index

		:return: loss(algorithm(w, x[i]), y[i])
		"""
		return self.loss(self._algorithm(index), self.sample[index].y)

	def _gd_step(self, index):
		""" w - learning_rate * gradient_descent """
		g = self._gradient(index)
		return np.array(self.weights) - [self.rate_learning * gd_i for gd_i in g]

	def _get_precedent_pos(self):
		return np.random.randint(len(self.sample))

	def _gradient(self, index):
		pass

	def algorithm(self, weights, x):
		pass

	def loss(self, y1, y2):
		""" Loss function (error) """
		pass

	def quality_diff(self):
		diff = np.sum(self.quality - self._prev_quality)
		return diff

	def weights_diff(self):
		diff = np.sum(self.weights - self._prev_weights)
		return diff

	def is_stop_iterating(self):
		return self.iteration > (2 * len(self.sample))

	def stability_quality(self):
		return self.quality_diff() - self._prev_quality_diff

	def is_stable_quality(self):
		return self.precision_quality >= abs(self.stability_quality())

	def is_stable_weights(self):
		return self.precision_weights >= abs(self.weights_diff())

	def is_stop_calculating(self):
		# Q is stable
		# and/or
		# weights stopped changing
		result = False
		if 1 < self.iteration:
			if self.is_stable_quality() and self.is_stable_weights() \
					or self.is_stop_iterating():
				result = True
		return result

	def calculate(self):
		def is_quality_overflow():
			return np.isinf(self.quality).any() or np.isnan(self.quality).any()

		def is_weights_overflow():
			return np.isinf(self.weights).any() or np.isnan(self.weights).any()

		while not self.is_stop_calculating():
			self.iteration += 1
			self._prev_quality_diff = self.quality_diff()
			pos = self._get_precedent_pos()
			try:
				self._calc_step(pos)
				if is_quality_overflow():
					raise ArithmeticError('quality overflow')
				if is_weights_overflow():
					raise ArithmeticError('weights overflow')
			except ArithmeticError as error:
				print('\t!ERROR!', error, 'at iteration:', self.iteration)
				break
		return self.weights

	def _calc_step(self, index):
		self.errors[index] = self._loss(index)

		self._set_weights(self._gd_step(index))
		# (1-rf)*Q+rf*loss
		self._set_quality(np.dot(1 - self.rate_forgetting, self.quality) + self.rate_forgetting * self.errors[index])
		# what is quality (q) -- function or value?
		return index  # for logging


def loss_biquad(y1, y2):
	"""
	Biquad

	:return: (y1 - y2) ** 2
	"""
	return (y1 - y2) ** 2


def loss_biquad_deriv(y1, y2):
	"""
	Biquad loss derivative by y1

	:return: 2 * (y1 - y2)
	"""
	return 2 * (y1 - y2)


class Default(Base):
	def loss(self, y1, y2):
		return loss_biquad(y1, y2)

	def loss_deriv(self, y1, y2):
		return loss_biquad_deriv(y1, y2)

	def info(self):
		print('Iteration:', self.iteration)
		if self.is_stop_iterating():
			print('\ttoo much iterations:', self.is_stop_iterating())
		print('\tquality is stable:', self.is_stable_quality(), self.stability_quality())
		print('\tweights stopped changing:', self.is_stable_weights(), self.weights_diff())

		print('quality:', self.quality)
		print('weights:', self.weights)
		print('errors:', list(self.errors))


class Simple(Default):
	"""
	Stochastic Gradient

	realisation 1
	"""

	def algorithm(self, weights, x):
		"""
		Linear

		:return: <w, x>
		:rtype: float
		"""
		result = np.dot(weights, x)
		return result

	def _gradient(self, index):
		# {'}loss * x[i] * y[i]
		alg = self._algorithm(index)
		loss = self.loss_deriv(alg, self.sample[index].y)
		xy = [x_i * self.sample[index].y for x_i in self.sample[index].x]
		lxy = [loss * xy_i for xy_i in xy]
		return lxy


class Activ(Default):
	"""
	Stochastic Gradient

	realisation 2

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

	def activate_deriv(self, z):
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
		return result

	def _gradient(self, index):
		# diff
		# {'a}loss * {'}activate(<w, x[i]>) * x[i]
		alg = self._algorithm(index)
		loss = self.loss_deriv(alg, self.sample[index].y)
		wx_vm = np.dot(self.weights, self.sample[index].x)
		act = self.activate_deriv(wx_vm)
		la = loss * act
		lax = [la * x_i for x_i in self.sample[index].x]
		return lax


def rate_len(n):
	"""
	:return: 1/n
	"""
	return 1 / n


def weights_init_rand(n):
	"""
	Requires normalisation of sample!

	:return: rand(-1/n, 1/n)
	"""
	result = np.random.uniform(-1 / n, 1 / n, n)
	return result


def loss_binary(y1, y2):
	"""
	http://www.machinelearning.ru/wiki/images/5/53/Voron-ML-Lin-SG.pdf

	:return: [y1 * y2 < 0]
	"""
	result = y1 * y2
	return result if 0 > result else 0


def loss_binary_approx(y1, y2):
	"""
	derivative by y1: y2

	:return: y1 * y2
	"""
	return y1 * y2
