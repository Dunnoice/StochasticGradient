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

	def __new__(cls, value, add_const_attr=False):
		"""
		:type value: list[float] | numpy.ndarray[float]
		:param add_const_attr: x[0] = artificial constant attribute (default=1),
		x starts from 1 in order to multiply to weights
		"""
		values = []
		for sample_i in value:
			i_y = len(sample_i) - 1
			x_i = sample_i[0:i_y]
			if add_const_attr:
				x_i = np.append(1., x_i)
			values.append(Precedent(x_i, sample_i[i_y]))
		return super().__new__(cls, values)


class Base:
	sample: Sample
	rate_learning: float
	rate_forgetting: float
	weights: [float]
	quality: float
	errors: [float]  # set?
	precision_weights: float
	precision_quality: float

	def __init__(self, sample, learning_rate, forgetting_rate, quality_precision, weights_precision=0):
		"""
		:param Sample sample: [[x ... , y]]
		:param forgetting_rate: functional smoothing
		"""
		self.sample = sample
		self.rate_learning = learning_rate
		self.rate_forgetting = forgetting_rate

		self.precision_weights = weights_precision
		self.precision_quality = quality_precision

		self.weights = self._init_weights()
		self._prev_weights = 0
		self.errors = np.zeros(len(self.sample))
		self.quality = self._init_quality()
		self._prev_quality = 0

	def _init_quality(self):
		"""
		Empirical risk

		assessment of functional

		loss of sample

		:rtype: float
		"""
		for i in range(len(self.sample)):
			self.errors[i] = self._loss(i)
		return np.sum(self.errors)

	def _init_weights(self):
		"""
		:rtype: list[float]
		"""
		precedent_length = len(self.sample[0].x)
		shape = precedent_length

		result = np.zeros(shape)
		# result = np.random.uniform(-1 / precedent_length, 1 / precedent_length, shape)  # requires normalisation
		# result = np.full(shape, 0.0001)
		return result

	def _set_weights(self, new_weights):
		self._prev_weights = self.weights
		self.weights = new_weights

	def _set_quality(self, new_quality):
		self._prev_quality = self.quality
		self.quality = new_quality

	def _get_precedent_pos(self):
		return np.random.randint(len(self.sample))

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
		gd = self._gradient_descent(index)
		return np.array(self.weights) - [self.rate_learning * gd_i for gd_i in gd]

	def _gradient_descent(self, index):
		pass

	def algorithm(self, weights, x):
		pass

	def loss(self, y1, y2):
		pass

	def quality_diff(self):
		diff = np.sum(self.quality - self._prev_quality)
		return diff

	def weights_diff(self):
		diff = np.sum(self.weights - self._prev_weights)
		return diff

	def is_stop_calculating(self, iteration, previous_q_diff):
		# Q is stable
		# and/or
		# weights stopped changing
		result = False

		def too_much_iterations():
			return iteration > (2 * len(self.sample))

		q_diff = self.quality_diff() - previous_q_diff
		w_diff = self.weights_diff()
		is_q_stable = self.precision_quality >= abs(q_diff)
		is_w_stable = self.precision_weights >= abs(w_diff)
		if (iteration > 10) and (is_q_stable and is_w_stable or too_much_iterations()):
			result = True
			print('\n Stop:', iteration)
			if too_much_iterations():
				print('\ttoo much iterations:', too_much_iterations())
			print('\tquality is stable:', is_q_stable, q_diff)
			print('\tweights stopped changing:', is_w_stable, w_diff)
		return result

	def calculate(self):
		i = 1
		prev_q_diff = 0
		while not self.is_stop_calculating(i, prev_q_diff):
			prev_q_diff = self.quality_diff()
			try:
				self._calc_step()
			except ArithmeticError as error:
				print('\t!ERROR!', error, 'at iteration:', i)
				break
			i += 1
		return i, self.weights

	def _calc_step(self):
		def is_quality_overflow():
			return np.isinf(self.quality).any() or np.isnan(self.quality).any()

		def is_weights_overflow():
			return np.isinf(self.weights).any() or np.isnan(self.weights).any()

		pos = self._get_precedent_pos()
		self.errors[pos] = self._loss(pos)

		self._set_weights(self._gd_step(pos))
		# (1-rf)*Q+rf*loss
		self._set_quality(np.dot(1 - self.rate_forgetting, self.quality) + self.rate_forgetting * self.errors[pos])

		if is_quality_overflow():
			raise ArithmeticError('quality overflow')
		if is_weights_overflow():
			raise ArithmeticError('weights overflow')

		return pos


def loss_bicubic(y1, y2=1):
	"""
	Loss function (error), bicubic

	:return: (y1 - y2) ** 2
	"""
	return (y1 - y2) ** 2


def loss_bicubic_deriv(y1, y2=1):
	"""
	bicubic loss derivative by y1

	:return: 2 * (y1 - y2)
	"""
	return 2 * (y1 - y2)


class Default(Base):
	def loss(self, y1, y2=1):
		return loss_bicubic(y1, y2)

	def loss_deriv(self, y1, y2=1):
		return loss_bicubic_deriv(y1, y2)

	def info(self):
		print('quality:', self.quality)
		print('weights:', self.weights)
		print('errors:', list(self.errors))


class Vect(Default):
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
		return result

	def _gradient_descent(self, index):
		# {'}loss * x[i] * y[i]
		alg = self._algorithm(index)
		loss = self.loss_deriv(alg, self.sample[index].y)
		xy = [x_i * self.sample[index].y for x_i in self.sample[index].x]
		lxy = [loss * xy_i for xy_i in xy]
		return lxy


class Deriv(Default):
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

	def _gradient_descent(self, index):
		# diff
		# {'a}loss * {'}activate(<w, x[i]>) * x[i]
		alg = self._algorithm(index)
		loss = self.loss_deriv(alg, self.sample[index].y)
		wx_vm = np.dot(self.weights, self.sample[index].x)
		act = self.activate_deriv(wx_vm)
		la = loss * act
		lax = [la * x_i for x_i in self.sample[index].x]
		return lax


def rate_forgetting_len(sample):
	return 1 / len(sample)
