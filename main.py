import numpy
import random
import collections

numpy.random.seed(0)


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
	errors: [float]  # TODO set?
	precision_weights: float
	precision_quality: float

	def __init__(self, sample, learning_rate, forgetting_rate, weights_precision, quality_precision, *, algorithm=None):
		"""
		:param Sample sample: [[x ... , y]]
		:param forgetting_rate: functional smoothing
		"""
		self.sample = sample
		self.rate_learning = learning_rate
		self.rate_forgetting = forgetting_rate

		if algorithm is not None:
			self.algorithm = algorithm
		self.precision_weights = weights_precision
		self.precision_quality = quality_precision

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
		# result = numpy.random.uniform(-1 / precedent_length, 1 / precedent_length, shape)  # requires normalisation
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
			result += self._loss(i)
		return result

	def _get_precedent_pos(self):
		return random.randrange(len(self.sample))

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

	def algorithm(self, weights, x):
		"""
		:return: <w, x>
		:rtype: float
		"""
		return round(numpy.dot(weights, x))

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
		# w - learning_rate * {'}loss * x[i] * y[i]
		alg = self._algorithm(index)
		loss = self.loss_diff(alg, self.sample[index].y)
		lrl = self.rate_learning * loss
		xy = [x_i * self.sample[index].y for x_i in self.sample[index].x]
		lrlxy = [lrl * xy_i for xy_i in xy]
		return numpy.array(self.weights) - lrlxy

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
			self.errors[pos] = self._loss(pos)
			try:
				self.weights.set(self._gradient_descent(pos))
				# (1-rf)*Q+rf*loss
				self.quality[0] = numpy.dot(1 - self.rate_forgetting, self.quality[0]) \
								  + self.rate_forgetting * self.errors[pos]
				if is_quality_overflow():
					raise ArithmeticError('quality overflow at iteration: ' + str(i))
				if is_weights_overflow():
					raise ArithmeticError('weights overflow at iteration: ' + str(i))
			except ArithmeticError as error:
				print('\t!ERROR!', error)
				break
			i += 1
		return i, self.weights


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

	def algorithm(self, weights, x):
		"""
		Applies activation function to x and weights

		:return: activate(sum(w[j] * x[j] - w[0], 1, len(x)))
		:rtype: float
		"""
		result = 0
		for j in range(len(x)):
			result += weights[j] * x[j] - weights[0]
		return round(self.activate(result))

	def _gradient_descent(self, index):
		# diff
		# w - learning_rate * {'a}loss * {'}activate(<w, x[i]>) * x[i]
		loss = self.loss_diff(self._algorithm(index), self.sample[index].y)
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


def forgetting_rate(sample):
	return 1 / len(sample)


file1 = 'samples/Wine Quality Data Set/winequality-red.csv'

dataset1 = numpy.loadtxt(file1, delimiter=';', skiprows=1)
names1 = numpy.genfromtxt(file1, delimiter=';', dtype=str, max_rows=1)

rate_learn = 1e-4
rate_forget = forgetting_rate(dataset1)
precision_w = 5e-2
precision_q = 3

options1 = {
	'sample': Sample(dataset1),
	'learning_rate': rate_learn,
	'forgetting_rate': rate_forget,
	'weights_precision': precision_w,
	'quality_precision': precision_q
}

sg1 = SG_simplified(**options1)
print('Calculate:', sg1.calculate())
info(sg1)

options2 = dict(options1)
options2['sample'] = Sample(dataset1, add_const_attr=True)

sg2 = SG(**options2)
print('Calculate:', sg2.calculate())
info(sg2)


import matplotlib.pyplot as plt

x = numpy.array([precedent.x for precedent in options1['sample']])
y = numpy.array([precedent.y for precedent in options1['sample']])

sg2y = numpy.array([sg2.algorithm(sg2.weights, precedent.x) for precedent in options1['sample']])

# plt.plot(main.dataset1, '.', alpha=0.3)
plt.plot(y, 'r*', sg2y, 'b.', alpha=0.1)

plt.show()
