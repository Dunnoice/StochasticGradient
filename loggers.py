class Logger:
	"""
	Saves log of previous values of watched variable -- loggable

	! Do update before changing value
	"""

	def _watch(self, loggable):
		self._loggable_ = [loggable]

	def __init__(self, loggable, f_conversion, empty_value=None):
		"""
		:param f_conversion: way to represent value in log; ex. type(object) or copy(object)
		"""
		self._log = []
		self._watch(loggable)
		self._empty_value = empty_value
		self._copy = f_conversion

	def _eq_loggable(self, value):
		""" Return self._loggable == value. """
		return self._loggable == value

	@property
	def _loggable(self):
		return self._copy(self._loggable_[0])

	@property
	def log(self):
		return self._log.copy()

	@property
	def logr(self):
		result = self.log
		result.reverse()
		return result

	@property
	def prev(self):
		""" Get last value """
		logr = self.logr
		return logr[0] if 0 < len(logr) else self._empty_value

	def _update(self):
		""" Push current value into log """
		self._log.append(self._loggable)

	def update_mutable(self):
		"""
		update log, if loggable's changed

		for mutable
		"""
		if 0 == len(self._log) or not self._eq_loggable(self.logr[0]):
			self._update()

	def update_immutable(self, value):
		"""
		If value is different from loggable, update and watch it instead

		for immutable
		"""
		if not self._eq_loggable(value):
			self._update()
			self._watch(value)


class ValueL:
	def __init__(self, value, type, *, empty_value=None):
		self.value = value
		self._logger = Logger(self.value, type, empty_value)

	@property
	def log(self):
		return self._logger.log

	@property
	def logr(self):
		result = self._logger.log
		result.reverse()
		return result

	@property
	def prev(self):
		return self._logger._copy(self._logger.prev)

	def set(self, value):
		self.value = value
		self._logger.update_immutable(self.value)


class ListL(list):
	"""	List that has log of its previous values """

	def __init__(self, seq=()):
		super().__init__(seq)
		self._logger = Logger(self, list, [])  # to protect update function

	def __setitem__(self, key, value):
		self._logger.update_mutable()
		super().__setitem__(key, value)

	@property
	def log(self):
		return self._logger.log

	@property
	def logr(self):
		return self._logger.logr

	@property
	def prev(self):
		return self._logger.prev

	def set(self, iterable):
		""" clear and extend """
		self._logger.update_mutable()
		super().clear()
		super().extend(iterable)

	def append(self, object):
		self._logger.update_mutable()
		super().append(object)

	def clear(self):
		self._logger.update_mutable()
		super().clear()

	def extend(self, iterable):
		self._logger.update_mutable()
		super().extend(iterable)

	def insert(self, index, object):
		self._logger.update_mutable()
		super().insert(index, object)

	def pop(self, *args, **kwargs):
		self._logger.update_mutable()
		super().pop(*args, **kwargs)

	def remove(self, object):
		self._logger.update_mutable()
		super().remove(object)

	def reverse(self):
		self._logger.update_mutable()
		super().reverse()

	def sort(self, *args, **kwargs):
		self._logger.update_mutable()
		super().sort(*args, **kwargs)
