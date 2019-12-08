# import sympy as sy


# a, x, y, i = sy.symbols('a x y i')

# w = sy.Indexed('w', i)
# w0 = 1
# a = sy.Sum(w - w0, (i, 1, 5))

# a = p
# a = p(w * (x ** i) - w0)
# a = w * (x ** i) - w0
# L = (a - y)**2

# print(sy.diff(L, a))

# class Sample(tuple):
# 	"""
# 	Formatted for readability sample
#
# 	starts from 1
# 	"""
#
# 	def __new__(cls, value):
# 		"""
# 		Formats sample to readable view
#
# 		starts from 1
#
# 		:type value: list[list[int|float, float]]
# 		"""
# 		values = []
# 		for sample_i in value:
# 			values.append({'x': sample_i[0], 'y': sample_i[1]})
# 		return super().__new__(cls, values)
#
# 	def __getitem__(self, key):
# 		print('key:', key)
# 		if type(key) is int:
# 			if 0 <= key:
# 				return super().__getitem__(key - 1)
# 		elif type(key) is slice:
# 			if key.start is not None:
# 				if 0 < key.start:
# 					return super().__getitem__(slice(key.start - 1, key.stop, key.step))
# 		else:
# 			raise ValueError(f'Sample cannot be indexed with values of type {type(key)}')
# 		return super().__getitem__(key)

# class ListLoggable(list):
# 	"""	List that can be created of required length and has log of previous values """
# 	def __init__(self, seq=(), /, length=0):
# 		super().__init__(seq)
# 		self._enlarge(length - len(self))
# 		self.log = ()
# 		if not seq:
# 			self._log_update()
#
# 	def __setitem__(self, key, value):
# 		self._log_update()
# 		super().__setitem__(key, value)
#
# 	def _log_update(self):
# 		""" Push current value into log """
# 		self.log = self.log + (super().copy(),)
#
# 	def _enlarge(self, number):
# 		if 0 < number:
# 			super().extend([None] * number)
#
# 	def append(self, object):
# 		self._log_update()
# 		super().append(object)
#
# 	def clear(self):
# 		self._log_update()
# 		super().clear()
#
# 	def extend(self, iterable):
# 		self._log_update()
# 		super().extend(iterable)
#
# 	def insert(self, index, object):
# 		self._log_update()
# 		super().insert(index, object)
#
# 	def set(self, iterable):
# 		""" clear() and extend() """
# 		self._log_update()
# 		super().clear()
# 		super().extend(iterable)
#
# 	def pop(self, *args, **kwargs):
# 		self._log_update()
# 		super().pop(*args, **kwargs)
#
# 	def remove(self, object):
# 		self._log_update()
# 		super().remove(object)
#
# 	def reverse(self):
# 		self._log_update()
# 		super().reverse()
#
# 	def sort(self, *args, **kwargs):
# 		self._log_update()
# 		super().sort(*args, **kwargs)

# import urllib.request as urlr
#
# url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# raw_data_1 = urlr.urlopen(url1)
# dataset1 = numpy.loadtxt(raw_data_1, delimiter=";", skiprows=1)

import matplotlib.pyplot as plt
import main

sample1 = main.Sample(main.dataset1)

x = main.numpy.array([precedent.x for precedent in sample1])
y = main.numpy.array([precedent.y for precedent in sample1])

sg2y = main.numpy.array([main.sg2.algorithm(main.sg2.weights, precedent.x) for precedent in sample1])

# plt.plot(main.dataset1, '.', alpha=0.3)
plt.plot(y, 'r.', alpha=0.1)

plt.show()
