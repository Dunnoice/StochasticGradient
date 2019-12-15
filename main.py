import numpy as np

import stochastic_gradient as SG

np.random.seed(0)


class SGL(SG.Default):
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
			cond = (qs, ws, 1)
		elif ws:
			cond = (qs, ws, 2)
		else:
			cond = (qs, ws)
		self.log_cond.append(cond)

	def info(self):
		super().info()
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


class SGDerivL(SGL, SG.Deriv):
	pass


class MyDerivLround(SGDerivL):
	def algorithm(self, weights, x):
		alg = super().algorithm(weights, x)
		return round(alg)


file1 = 'samples/Wine Quality Data Set/winequality-red.csv'

dataset1 = np.loadtxt(file1, delimiter=';', skiprows=1)
names1 = np.genfromtxt(file1, delimiter=';', dtype=str, max_rows=1)

options1_diff = {
	'sample': SG.Sample(dataset1, add_const_attr=True),
	'learning_rate': 1e-5,
	'forgetting_rate': 1e-4,
	'quality_precision': 5e-5,
	'weights_precision': 0,
}

sgdl1 = MyDerivLround(**options1_diff)
print('Calculate:', sgdl1.calculate())
sgdl1.info()
sg1y = np.array([sgdl1.algorithm(sgdl1.weights, precedent.x) for precedent in options1_diff['sample']])
print('test 1:', sg1y)

options2_diff = {
	'sample': SG.Sample(dataset1, add_const_attr=True),
	'learning_rate': 1e-5,
	'forgetting_rate': 1e-4,
	'quality_precision': 1e-4,
	'weights_precision': 5e-5,
}

sgdl2 = SGDerivL(**options2_diff)
print('Calculate:', sgdl2.calculate())
sgdl2.info()
sg2y = np.array([sgdl2.algorithm(sgdl2.weights, precedent.x) for precedent in options2_diff['sample']])
print('test 2:', sg2y)

# import matplotlib.pyplot as plt

# x = np.array([precedent.x for precedent in options1_diff['sample']])
# y = np.array([precedent.y for precedent in options1_diff['sample']])
#
# # plt.plot(y, 'r*', sg1y, 'b.', alpha=0.1)
# plt.plot(sg1y, 'b.', alpha=0.1)
#
# # plt.show()
