import numpy as np

import stochastic_gradient as sg


class SGL(sg.Default):
	def __init__(self, sample, learning_rate, forgetting_rate, quality_precision, weights_precision):
		super().__init__(sample, learning_rate, forgetting_rate, quality_precision, weights_precision)
		self.log_w, self.log_q, self.log_p, self.log_wd, self.log_qd, self.log_qs = [], [], [], [], [], []
		self.log_cond = []

	def _set_weights(self, new_weights):
		self.log_w.append(list(self.weights))
		super()._set_weights(new_weights)

	def _set_quality(self, new_quality):
		self.log_q.append(self.quality)
		super()._set_quality(new_quality)

	def _calc_step(self):
		qd_prev = super().quality_diff()
		self.log_p.append(super()._calc_step())  # important
		self.log_qd.append(self.quality_diff())
		self.log_qs.append(self.quality_diff() - qd_prev)
		self.log_wd.append(self.weights_diff())
		qs = self.precision_quality >= abs(super().quality_diff() - qd_prev)
		ws = self.precision_weights >= abs(super().weights_diff())
		if qs:
			cond = 'q'
		elif ws:
			cond = 'w'
		else:
			cond = ''
		self.log_cond.append(cond)

	def info_log(self):
		rlog_q = self.log_q.copy()
		rlog_q.reverse()
		print('rlog_q:', rlog_q)
		rlog_w = self.log_w.copy()
		rlog_w.reverse()
		print('rlog_w:', rlog_w)

		print('log_qs:', self.log_qs)
		print('log_qd:', self.log_qd)
		print('log_wd:', self.log_wd)
		print('log_cond:', self.log_cond)

		print('log_p:', self.log_p)


class SGLDeriv(SGL, sg.Deriv):
	pass


class SGLVect(SGL, sg.Vect):
	pass



random_seed = 0
file1 = 'samples/Wine Quality Data Set/winequality-red.csv'

dataset1 = np.loadtxt(file1, delimiter=';', skiprows=1)
names1 = np.genfromtxt(file1, delimiter=';', dtype=str, max_rows=1)


class MyDerivLround(SGLDeriv):
	def algorithm(self, weights, x):
		alg = super().algorithm(weights, x)
		return round(alg)


np.random.seed(random_seed)
options1_diff = dict(
	sample=sg.Sample(dataset1, add_const_attr=True),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=5e-5,
	weights_precision=0,
)

sgld1 = MyDerivLround(**options1_diff)
print('Calculate SGderiv_round:\n', sgld1.calculate())
sgld1.info()
# sgld1.info_log()


np.random.seed(random_seed)
options1_diff2 = dict(
	sample=sg.Sample(dataset1, add_const_attr=True),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=9e-6,
	weights_precision=5e-5,
)

sgld1_2 = SGLDeriv(**options1_diff2)
print('Calculate SGderiv:\n', sgld1_2.calculate())
sgld1_2.info()
sgld1_2.info_log()


np.random.seed(random_seed)
options1_vect = dict(
	sample=sg.Sample(dataset1),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=5e-5,
	weights_precision=0,
)


class MyVectLround(SGLVect):
	def algorithm(self, weights, x):
		alg = super().algorithm(weights, x)
		return round(alg)


sglv1 = MyVectLround(**options1_vect)
print('Calculate SGvect_round:\n', sglv1.calculate())
sglv1.info()
# sglv1.info_log()
