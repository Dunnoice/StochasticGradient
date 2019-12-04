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
	"""

	def __new__(cls, value, add_const_attr=False):
		"""
		:type value: list[list[float]]
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
		self.quality = ListLoggable([None])
		self.quality[0] = self._init_quality()
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

		def q_is_stable():
			return 1.e+3 > numpy.sum(q_prev - q)

		def weights_stopped_changing():
			return 0 == numpy.sum(w_prev - w)

		def too_much_iterations():
			return iteration > (1.5 * len(self.sample))

		if (iteration > 1) and q_is_stable() and weights_stopped_changing() or too_much_iterations():
			print('Reason for stop:')
			print('\tquality is stable:', q_is_stable())
			print('\tweights stopped changing:', weights_stopped_changing())
			print('\tOR too much iterations:', too_much_iterations(), iteration)
			result = True
		return result

	def calculate(self):
		def is_overflow():
			return numpy.isinf(self.weights).any() or numpy.isnan(self.weights).any() \
				   or numpy.isinf(self.quality[0]).any() or numpy.isnan(self.quality[0]).any()

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

			if is_overflow():
				# raise ArithmeticError('Weights overflow at iteration: ' + str(i))
				print('***ERROR***', 'Overflow at iteration:', i)
				i = 10000

			i += 1

		return [i, self.quality.prev(), self.weights.prev()]

	def algorithm(self, index):
		"""
		:return: <w, x>
		:rtype: float
		"""
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
		loss = self.loss_diff(self.algorithm(index), self.sample[index].y)
		lrl = self.rate_learning * loss
		lrlx = [lrl * x_i for x_i in self.sample[index].x]
		lrlxy = lrlx * self.sample[index].y
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


test_sample = [
	[1., 2., 0.5],
	[3., 4., 0.5],
	[5., 6., 0.5]
]

learning_rate = 0.05


def forgetting_rate(sample):
	return 1 / len(sample)


import urllib.request as urlr

url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
raw_data_1 = urlr.urlopen(url1)
dataset1 = numpy.loadtxt(raw_data_1, delimiter=";", skiprows=1)

sg1 = SG(Sample(dataset1), learning_rate, forgetting_rate(dataset1))
print('Calculate:', sg1.calculate())
print('weights:', sg1.weights)
print('w.log:', sg1.weights.log)
print('quality:', sg1.quality)
print('q.log:', sg1.quality.log)

sg2 = SG(Sample(dataset1, add_const_attr=True), learning_rate, forgetting_rate(dataset1))
print('Calculate:', sg2.calculate())
print('weights:', sg2.weights)
print('w.log:', sg2.weights.log)
print('quality:', sg2.quality)
print('q.log:', sg2.quality.log)
