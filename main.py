import copy
import random
import numpy
import collections


class Precedent(collections.namedtuple('Precedent', 'x y')):
	@property
	def value(self):
		return numpy.array([self.x, self.y])


class Sample(tuple):
	"""
	Formatted for readability sample
	"""

	def __new__(cls, value):
		"""
		:type value: list[list[list[float], float]]
		"""
		values = []
		for sample_i in value:
			values.append(Precedent(sample_i[0], sample_i[1]))
		return super().__new__(cls, values)


class ListLoggable(list):
	"""	List that has log of its previous values """

	def __init__(self, seq=()):
		super().__init__(seq)
		self.log = ()
		if numpy.array(seq).all():
			self._log_update()

	def __setitem__(self, key, value):
		self._log_update()
		super().__setitem__(key, value)

	def _log_update(self):
		""" Push current value into log """
		self.log = self.log + (super().copy(),)

	def set(self, iterable):
		""" clear() and extend() """
		self._log_update()
		super().clear()
		super().extend(iterable)

	def append(self, object):
		self._log_update()
		super().append(object)

	def clear(self):
		self._log_update()
		super().clear()

	def extend(self, iterable):
		self._log_update()
		super().extend(iterable)

	def insert(self, index, object):
		self._log_update()
		super().insert(index, object)

	def pop(self, *args, **kwargs):
		self._log_update()
		super().pop(*args, **kwargs)

	def remove(self, object):
		self._log_update()
		super().remove(object)

	def reverse(self):
		self._log_update()
		super().reverse()

	def sort(self, *args, **kwargs):
		self._log_update()
		super().sort(*args, **kwargs)


class SG:
	"""
	Stochastic Gradient
	"""
	sample: Sample
	rate_learning: float
	rate_forgetting: float
	weights: [float]
	decision_threshold: float
	quality: float
	errors: [float]

	def __init__(self, sample, learning_rate, forgetting_rate):
		"""
		:param sample: [[x[x1, x2], y]]
		:param forgetting_rate: functional smoothing
		"""
		self.sample = Sample(sample)
		self.rate_learning = learning_rate
		self.rate_forgetting = forgetting_rate

		self.weights = ListLoggable(self._init_weights())
		self.quality = ListLoggable([None])
		self.quality[0] = self._init_quality()
		self.errors = ListLoggable(numpy.empty_like(self.sample))  # list[list[float]]

	def _init_weights(self):
		"""
		:rtype: list[float]
		"""
		precedent_length = len(self.sample[1].x)
		shape = precedent_length + 1
		# result = numpy.zeros(shape)
		# result = numpy.random.uniform(-1 / precedent_length, 1 / precedent_length, shape)
		result = numpy.full(shape, 0.0001)

		self.decision_threshold = result[0]
		result = result[1:]
		return result

	def _init_quality(self):
		"""
		Empirical risk

		assessment of functional

		loss of sample

		:rtype: float
		"""
		result = 0
		for i in range(len(self.sample)):
			result += self.precedent_loss(i)
		return result

	def _get_precedent_pos(self):
		return random.randrange(len(self.sample))

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
		if (iteration > 1) and (1.e+3 > numpy.sum(q_prev - q)) and (0 == numpy.sum(w_prev - w)) or (
				iteration > len(sample) ** 2):
			result = True
		return result

	def calculate(self):
		i = 1
		while not self.is_stop_calculating(self.quality.log[len(self.quality.log) - 1],
										   self.weights.log[len(self.weights.log) - 1], i):
			pos = self._get_precedent_pos()
			precedent = self.sample[pos]
			self.errors[pos] = self.precedent_loss(pos)
			self.weights.set(self.gradient_descent_step(pos))
			# (1-rf)*Q+rf*loss
			self.quality[0] = numpy.dot(1 - self.rate_forgetting, self.quality[0]) + self.rate_forgetting * self.errors[
				pos]
			i += 1

		return i

	def activate(self, x):
		"""
		Activation function

		phi

		scalar from x

		:param x: list[float] | float
		:return: x
		"""
		result = 0
		if type(x) is list:
			for x_i in x:
				result += x_i
		else:
			result = x
		return result

	def diff_activate(self, x):
		"""
		activate derivative

		:return: 1
		"""
		return 1

	def algorithm(self, index):
		"""
		Applies activation function to x and weights

		:return: activate(sum(w[j] * x[j] - w[last], 1, len(x)))
		:rtype: float
		"""
		result = 0
		for j in range(len(self.sample[index].x)):
			result += self.weights[j] * self.sample[index].x[j] - self.decision_threshold
		return self.activate(result)

	def precedent_loss(self, index):
		""" Calculates loss for precedent on index """
		return self.loss(self.algorithm(index), self.sample[index].y)

	def loss(self, y1, y2):
		"""
		Loss function

		error

		bicubic

		:return: (y1 - y2) ** 2
		"""
		return (y1 - y2) ** 2

	def diff_loss(self, y1, y2):
		"""
		loss derivative by y1

		:return: 2 * (y1- y2)
		"""
		return 2 * (y1 - y2)

	def gradient_descent_step(self, index):
		# w - learning_rate * {'a}loss * {'}activate(<w, x[i]>) * x[i]
		g1 = self.diff_loss(self.algorithm(index), self.sample[index].y)
		g2 = self.rate_learning * g1
		g3 = g1 * g2
		g4 = g3 * self.diff_activate(numpy.dot(self.weights, self.sample[index].x))
		g5 = [g4 * x_i for x_i in self.sample[index].x]
		g6 = numpy.array(self.weights) - g5  # TODO How to change decision threshold?
		return g6  # diff


sample = [
	[[1., 2.], 0.5],
	[[3., 4.], 0.5],
	[[5., 6.], 0.5]
]

sg1 = SG(sample, 1, 1 / len(sample))
print('Calculate:', sg1.calculate())
print('weights:', sg1.weights)
print('w.log:', sg1.weights.log)
print('quality:', sg1.quality)
print('q.log:', sg1.quality.log)
