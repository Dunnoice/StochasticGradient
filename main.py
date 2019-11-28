import copy
import random
import numpy
import collections


class Precedent(collections.namedtuple('Precedent', 'x y i')):
	@property
	def value(self):
		return numpy.array([self.x, self.y])


class Sample(tuple):
	"""
	Formatted for readability sample

	starts from 1
	"""

	def __new__(cls, value):
		"""
		Formats sample to readable view

		starts from 1

		:type value: list[list[list[float], float]]
		"""
		values = [None]
		i = 1
		for sample_i in value:
			values.append(Precedent(sample_i[0], sample_i[1], i))
			i += 1
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

	def __init__(self, sample, learning_rate, forgetting_rate):
		"""
		:param Sample sample: [x[x1, x2], y]
		:param forgetting_rate: functional smoothing
		"""
		# list[list[int|float, float]]
		self.sample = sample
		self.rate_learning = learning_rate
		self.rate_forgetting = forgetting_rate

		self.weights = ListLoggable(self._init_weights())
		# list[list[float]]
		self.quality = ListLoggable([self._init_quality()])
		# float
		self.errors = ListLoggable(numpy.empty_like(self.sample))
		# list[list[float]]

	def _init_weights(self):
		"""
		:rtype: list[list[float]]
		"""
		# only weights can start from 0!
		precedent_size = len(self.sample[1].x)
		shape = (len(self.sample), precedent_size)  # len(sample) is bigger to accommodate 0
		# result = numpy.zeros(shape)
		# result = numpy.random.uniform(-1 / precedent_size, 1 / precedent_size, shape)
		result = numpy.full(shape, 0.0001)
		return result

	def _init_quality(self):
		"""
		Empirical risk

		assessment of functional

		loss of sample

		:rtype: float
		"""
		result = 0
		for i in range(1, len(self.sample)):
			result += self.loss(self.sample[i])
		return result

	def _get_precedent_pos(self):
		return random.randint(1, len(self.sample) - 1)

	def get_precedent(self):
		return self.sample[self._get_precedent_pos()]

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
		if (iteration > 1) and (1.e+3 > numpy.sum(q_prev - q)) and (0 == numpy.sum(w_prev - w)) or (iteration > len(sample) ** 2):
			result = True
		return result

	def calculate(self):
		i = 1
		while not self.is_stop_calculating(self.quality.log[len(self.quality.log) - 1],
										   self.weights.log[len(self.weights.log) - 1], i):
			precedent = self.get_precedent()
			loss = self.errors[precedent.i] = self.loss(precedent)
			self.weights.set(self.gradient_descent(precedent))
			self.quality.set(self.calc_quality(loss))
			i += 1

		return i

	def activation(self, x):
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

	def diff_activation(self, x):
		"""
		activation derivative

		:return: 1
		"""
		return 1

	def algorithm(self, precedent):
		"""
		Applies activation function to x and weights

		:param Precedent precedent:
		:return: activation(sum(w[j] * x[j] - w[0], 1, len(x)))
		:rtype: float
		"""
		result = 0
		for j in range(1, len(self.sample)):  # correct: requires iteration from 1
			a1 = self.weights[precedent.i] * precedent.x - self.weights[0]
			result += self.activation(a1)
		return result

	def loss(self, precedent):
		"""
		Loss function

		error

		mathematically: L(a(w, x), y)

		difference between response of algorithm (a) and correct response in sample (y)

		bicubic

		:param Precedent precedent: [x, y]
		:return: (a(w, x) - y) ** 2
		"""
		return (self.algorithm(precedent) - precedent.y) ** 2

	def diff_loss(self, precedent):
		"""
		loss derivative by algorithm

		:param precedent: [x, y]
		:return: 2 * (a(w, x) - y)
		"""
		return 2 * (self.algorithm(precedent) - precedent.y)

	def gradient_descent_step(self, precedent):
		# w := w - learning_rate*{'a}loss*{'}activation(<w, x[i]>)*x[i]
		weights = self.weights[precedent.i]
		return (weights - self.rate_learning * self.diff_loss(precedent)
				* self.diff_activation(numpy.dot(weights, precedent.x)) * precedent.x)  # diff

	def gradient_descent(self, precedent):
		result = [None] * len(self.weights)
		for i in range(len(self.weights)):
			result[i] = self.gradient_descent_step(precedent)
		return result

	def calc_quality(self, loss):
		# (1-rf)*Q+rf*loss
		return numpy.dot(1 - self.rate_forgetting, self.quality) + self.rate_forgetting * loss


sample = Sample([
	[[1, 2], 0.5],
	[[3, 4], 0.5],
	[[5, 6], 0.5]
])

sg1 = SG(sample, 1, 1 / len(sample))
print('Calculate:', sg1.calculate())
print('weights:', sg1.weights)
print('w.log:', sg1.weights.log)
print('quality:', sg1.quality)
print('q.log:', sg1.quality.log)
