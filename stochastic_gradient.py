import numpy

import collections


class Precedent(collections.namedtuple('Precedent', 'x y')):
	@property
	def value(self):
		return numpy.array([self.x, self.y])


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
				x_i = numpy.append(1., x_i)
			values.append(Precedent(x_i, sample_i[i_y]))
		return super().__new__(cls, values)


class StochasticGradient:
	sample: Sample
	rate_learning: float
	rate_forgetting: float
	weights: [float]
	quality: float
	errors: [float]  # set?
	precision_weights: float
	precision_quality: float

	def __init__(self, sample, learning_rate, forgetting_rate, weights_precision, quality_precision):
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
		self.errors = numpy.zeros(len(self.sample))
		self.quality = self._init_quality()

	def _init_quality(self):
		"""
		Empirical risk

		assessment of functional

		loss of sample

		:rtype: float
		"""
		for i in range(len(self.sample)):
			self.errors[i] = self._loss(i)
		return numpy.sum(self.errors)

	def _init_weights(self):
		"""
		:rtype: list[float]
		"""
		precedent_length = len(self.sample[0].x)
		shape = precedent_length

		result = numpy.zeros(shape)
		# result = numpy.random.uniform(-1 / precedent_length, 1 / precedent_length, shape)  # requires normalisation
		# result = numpy.full(shape, 0.0001)
		return result

	def _get_precedent_pos(self):
		pass

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
		return numpy.array(self.weights) - [self.rate_learning * gd_i for gd_i in gd]

	def _gradient_descent(self, index):
		pass

	def algorithm(self, weights, x):
		pass

	def loss(self, y1, y2):
		pass

	def is_stable_weights(self, weights, weights_previous):
		difference = numpy.sum(weights) - numpy.sum(weights_previous)
		return self.precision_weights > abs(difference), difference

	def is_stable_quality(self, quality, quality_previous):
		difference = numpy.sum(quality) - numpy.sum(quality_previous)
		return self.precision_quality > abs(difference), difference

	def is_stop_calculating(self, quality_previous, weights_previous, iteration):
		# TODO finish condition & remove argument
		# Q is stable
		# and/or
		# weights stopped changing
		result = False
		q = numpy.array(self.quality)
		q_prev = numpy.array(quality_previous)
		w = numpy.array(self.weights)
		w_prev = numpy.array(weights_previous)

		def too_much_iterations():
			return iteration > (1.75 * len(self.sample))

		qs = self.is_stable_quality(q, q_prev)
		ws = self.is_stable_weights(w, w_prev)
		if (iteration > 10) and (qs[0] and ws[0] or too_much_iterations()):
			print('\n Reason for stop:')
			print('\tquality is stable:', qs)
			print('\tweights stopped changing:', ws)
			print('\tOR too much iterations:', too_much_iterations(), iteration)
			result = True
		return result

	def _calc_step(self):
		def is_quality_overflow():
			return numpy.isinf(self.quality).any() or numpy.isnan(self.quality).any()

		def is_weights_overflow():
			return numpy.isinf(self.weights).any() or numpy.isnan(self.weights).any()

		pos = self._get_precedent_pos()
		self.errors[pos] = self._loss(pos)

		self.weights = self._gd_step(pos)
		# (1-rf)*Q+rf*loss
		self.quality = numpy.dot(1 - self.rate_forgetting, self.quality) \
					   + self.rate_forgetting * self.errors[pos]

		if is_quality_overflow():
			raise ArithmeticError('quality overflow')
		if is_weights_overflow():
			raise ArithmeticError('weights overflow')

		return pos

	def calculate(self):
		prev_q = self.quality
		prev_w = self.weights
		i = 1
		while not self.is_stop_calculating(prev_q, prev_w, i):
			try:
				self._calc_step()
			except ArithmeticError as error:
				print('\t!ERROR!', error, 'at iteration:', i)
				break
			i += 1
		return i, self.weights


def forgetting_rate(sample):
	return 1 / len(sample)
