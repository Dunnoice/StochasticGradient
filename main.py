import numpy
import random
import collections


class ListLoggable(list):
	"""	List that has log of its previous values """

	def __init__(self, seq=()):
		super().__init__(seq)
		self._log = []
		# if numpy.array(seq).all():
		# 	self._log_update()

	def __setitem__(self, key, value):
		self._log_update()
		super().__setitem__(key, value)

	def _log_update(self):
		""" Push current value into log """
		self._log.append(super().copy())

	@property
	def log(self):
		return self._log.copy()

	@property
	def logr(self):
		result = self._log.copy()
		result.reverse()
		return result

	def set(self, iterable):
		""" clear() and extend() """
		self._log_update()
		super().clear()
		super().extend(iterable)

	def prev(self):
		""" Get last value """
		if 0 == len(self._log):
			return self._log
		return self._log[len(self._log) - 1]

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
	"""

	def __new__(cls, value, add_const_attr=False):
		"""
		:type value: list[list[float]] | numpy.ndarray[float]
		:param add_const_attr: x[0] = artificial constant attribute (default=1),
		x starts from 1 in order to multiply to weights
		"""
		values = []
		for sample_i in value:
			i_y = len(sample_i) - 1
			x_i = sample_i[0:i_y]
			if add_const_attr:
				x_i = [1.] + x_i
			values.append(Precedent(x_i, sample_i[i_y]))
		return super().__new__(cls, values)


class SG_simplified:
	"""
	Stochastic Gradient

	uses vector multiplication
	"""
	sample: Sample
	rate_learning: float
	rate_forgetting: float
	weights: [float]
	quality: float
	errors: [float]

	def __init__(self, sample, learning_rate, forgetting_rate):
		"""
		:param Sample sample: [[x ... , y]]
		:param forgetting_rate: functional smoothing
		"""
		self.sample = sample
		self.rate_learning = learning_rate
		self.rate_forgetting = forgetting_rate

		self.weights = ListLoggable(self._init_weights())
		self.quality = ListLoggable([self._init_quality()])  # due to lack of pointers value of quality is stored in [0]
		self.errors = ListLoggable(numpy.zeros(len(self.sample)))  # list[float]

	def _init_weights(self):
		"""
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
			result += self.loss_precedent(i)
		return result

	def _get_precedent_pos(self):
		return random.randrange(len(self.sample))

	def is_stable_quality(self, quality, quality_previous):
		difference = numpy.sum(quality) - numpy.sum(quality_previous)
		return 2. > difference

	def is_stable_weights(self, weights, weights_previous):
		difference = numpy.sum(weights) - numpy.sum(weights_previous)
		return 0.0001 > difference

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
		if (iteration > 10) and (qs and ws or too_much_iterations()):
			print('Reason for stop:')
			print(' quality is stable:', qs)
			print(' weights stopped changing:', ws)
			print(' OR too much iterations:', too_much_iterations(), iteration)
			result = True
		return result

	def calculate(self):
		def is_quality_overflow():
			return numpy.isinf(self.quality[0]).any() or numpy.isnan(self.quality[0]).any()

		def is_weights_overflow():
			return numpy.isinf(self.weights).any() or numpy.isnan(self.weights).any()

		i = 1
		while not self.is_stop_calculating(self.quality.prev(),
										   self.weights.prev(), i):
			pos = self._get_precedent_pos()
			# unclear: loss(<w, x[i]> y[i]) -- one or two args?
			self.errors[pos] = self.loss_precedent(pos)
			self.weights.set(self.gradient_descent(pos))
			# (1-rf)*Q+rf*loss
			self.quality[0] = numpy.dot(1 - self.rate_forgetting, self.quality[0]) \
							  + self.rate_forgetting * self.errors[pos]

			if is_quality_overflow():
				# raise ArithmeticError('overflow at iteration: ' + str(i))
				print('\t!ERROR! quality overflow at iteration:', i)
				i = 10000
			if is_weights_overflow():
				print('\t!ERROR! weights overflow at iteration:', i)
				i = 10000

			i += 1

		return (i, self.quality.prev(), self.weights.prev())

	def algorithm(self, index):
		"""
		:return: <w, x>
		:rtype: float
		"""
		test = numpy.multiply(self.weights, self.sample[index].x)
		test2 = numpy.sum(test)
		return numpy.dot(self.weights, self.sample[index].x)

	def loss(self, y1, y2=1):
		"""
		Loss function

		error

		bicubic

		:return: (y1 - y2) ** 2
		"""
		return (y1 - y2) ** 2

	def loss_precedent(self, index):
		"""
		Calculates loss of algorithm for precedent on index

		:return: loss(alg(w, x[i]), y[i])
		"""
		return self.loss(self.algorithm(index), self.sample[index].y)

	def loss_diff(self, y1, y2=1):
		"""
		loss derivative by y1

		:return: 2 * (y1 - y2)
		"""
		return 2 * (y1 - y2)

	def gradient_descent(self, index):
		# w - learning_rate * {'}loss * x[i] * y[i]
		alg = self.algorithm(index)
		loss = self.loss_diff(alg, self.sample[index].y)
		lrl = self.rate_learning * loss
		xy = [x_i * self.sample[index].y for x_i in self.sample[index].x]
		lrlxy = [lrl * xy_i for xy_i in xy]
		return numpy.array(self.weights) - lrlxy


class SG(SG_simplified):
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

	def gradient_descent(self, index):
		# diff
		# w - learning_rate * {'a}loss * {'}activate(<w, x[i]>) * x[i]
		loss = self.loss_diff(self.algorithm(index), self.sample[index].y)
		act = self.activate_diff(numpy.dot(self.weights, self.sample[index].x))
		lrla = self.rate_learning * loss * act
		lrlax = [lrla * x_i for x_i in self.sample[index].x]
		result = numpy.array(self.weights) - lrlax
		return result


def info(sg):
	print('quality:', sg.quality)
	print('weights:', sg.weights)
	print('q.logr:', sg.quality.logr)
	print('w.logr:', sg.weights.logr)


test_sample = [
	[1., 2., 0.5],
	[3., 4., 0.5],
	[5., 6., 0.5]
]

learning_r = 0.0001
forgetting_r = learning_r


def forgetting_rate(sample):
	return 1 / len(sample)


import urllib.request as urlr

url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
raw_data_1 = urlr.urlopen(url1)
dataset1 = numpy.loadtxt(raw_data_1, delimiter=";", skiprows=1)

sg1 = SG_simplified(Sample(dataset1), learning_r, forgetting_r)
print('Calculate:', sg1.calculate())
info(sg1)

sg2 = SG(Sample(dataset1, add_const_attr=True), learning_r, forgetting_r)
print('Calculate:', sg2.calculate())
info(sg2)
