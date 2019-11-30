import random
import numpy
import collections


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

	def prev(self):
		""" Get last value """
		if 0 == len(self.log):
			return self.log
		return self.log[len(self.log) - 1]

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


class Precedent(collections.namedtuple('Precedent', 'x y')):
	@property
	def value(self):
		return numpy.array([self.x, self.y])


class Sample(tuple):
	"""
	Formatted sample

	x starts from 1 in order to multiply to weights

	x[0] = artificial constant attribute
	"""

	def __new__(cls, value):
		"""
		:type value: list[list[float]]
		"""
		values = []
		for sample_i in value:
			i_y = len(sample_i) - 1
			values.append(Precedent([1.] + sample_i[0:i_y], sample_i[i_y]))
		return super().__new__(cls, values)


class SG:
	"""
	Stochastic Gradient
	"""
	sample: Sample
	rate_learning: float
	rate_forgetting: float
	weights: [float]
	quality: float
	errors: [float]

	def __init__(self, sample, learning_rate, forgetting_rate):
		"""
		:param sample: [[x ... , y]]
		:param forgetting_rate: functional smoothing
		"""
		self.sample = Sample(sample)
		self.rate_learning = learning_rate
		self.rate_forgetting = forgetting_rate

		self.weights = ListLoggable(self._init_weights())
		self.quality = ListLoggable([None])
		self.quality[0] = self._init_quality()
		self.errors = ListLoggable(numpy.zeros(len(self.sample)))  # list[float]

	def _init_weights(self):
		"""
		weights start from 1 in order to accommodate
		weights[0] = decision threshold

		:rtype: list[float]
		"""
		precedent_length = len(self.sample[0].x)
		shape = precedent_length

		result = numpy.zeros(shape)
		# result = numpy.random.uniform(-1 / precedent_length, 1 / precedent_length, shape)
		# result = numpy.full(shape, 0.0001)
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
		if (iteration > 1) and (1.e+3 > numpy.sum(q_prev - q)) and (0 == numpy.sum(w_prev - w)) \
				or (iteration > (2 * len(self.sample))):
			print('Reason for stop:')
			print('\tQuality is stable: 1.e+3 > numpy.sum(q_prev - q)', 1.e+3 > numpy.sum(q_prev - q))
			print('\tweights stopped changing: 0 == numpy.sum(w_prev - w)', 0 == numpy.sum(w_prev - w))
			print('\tOR iteration > (2 * len(self.sample))', iteration > (2 * len(self.sample)))
			result = True
		return result

	def calculate(self):
		i = 1
		while not self.is_stop_calculating(self.quality.prev(),
										   self.weights.prev(), i):
			pos = self._get_precedent_pos()
			self.errors[pos] = self.precedent_loss(pos)
			self.weights.set(self.gradient_descent(pos))
			# (1-rf)*Q+rf*loss
			self.quality[0] = numpy.dot(1 - self.rate_forgetting, self.quality[0]) \
							  + self.rate_forgetting * self.errors[pos]
			i += 1

		return i

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

	def diff_activate(self, z):
		"""
		activate derivative

		:return: 1
		"""
		return 1

	def algorithm(self, index):
		"""
		Applies activation function to x and weights

		:return: activate(sum(w[j] * x[j] - w[0], 1, len(x)))
		:rtype: float
		"""
		result = 0
		for j in range(len(self.sample[index].x)):
			result += self.weights[j] * self.sample[index].x[j] - self.weights[0]
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

	def gradient_descent(self, index):
		# w - learning_rate * {'a}loss * {'}activate(<w, x[i]>) * x[i]
		g1 = self.diff_loss(self.algorithm(index), self.sample[index].y)
		g2 = self.rate_learning * g1
		g3 = g1 * g2
		g4 = g3 * self.diff_activate(numpy.dot(self.weights, self.sample[index].x))
		g5 = [g4 * x_i for x_i in self.sample[index].x]
		g6 = numpy.array(self.weights) - g5
		return g6  # diff


sample_test = [
	[1., 2., 0.5],
	[3., 4., 0.5],
	[5., 6., 0.5]
]

import urllib.request as urlr

url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
raw_data_1 = urlr.urlopen(url1)
dataset1 = numpy.loadtxt(raw_data_1, delimiter=";", skiprows=1)

sg1 = SG(dataset1, 0.05, 1 / len(dataset1))
print('Calculate:', sg1.calculate())
print('weights:', sg1.weights)
print('w.log:', sg1.weights.log)
print('quality:', sg1.quality)
print('q.log:', sg1.quality.log)
