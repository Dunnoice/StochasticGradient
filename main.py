import numpy as np

import stochastic_gradient as SG

np.random.seed(0)


class SGDL(SG.Diff):
	def __init__(self, sample, learning_rate, forgetting_rate, weights_precision, quality_precision):
		super().__init__(sample, learning_rate, forgetting_rate, weights_precision, quality_precision)
		self.w_log, self.q_log, self.p_log, self.sw_log, self.sq_log = [], [], [], [], []

	def _calc_step(self):
		self.p_log.append(super()._calc_step())
		self.w_log.append(list(self.weights))
		self.q_log.append(self.quality)

	def is_stable_quality(self, quality, quality_previous):
		result = super().is_stable_quality(quality, quality_previous)
		self.sq_log.append(result[1])
		return result

	def is_stable_weights(self, weights, weights_previous):
		result = super().is_stable_weights(weights, weights_previous)
		self.sw_log.append(result[1])
		return result

	def algorithm(self, weights, x):
		return round(super().algorithm(weights, x))

	def info(self):
		super().info()
		q_log = self.q_log.copy()
		q_log.reverse()
		print('q.logr:', q_log)
		w_log = self.w_log.copy()
		w_log.reverse()
		print('w.logr:', w_log)
		print('sq_log', self.sq_log)
		print('sw_log', self.sw_log)
		print('p_log:', self.p_log)


file1 = 'samples/Wine Quality Data Set/winequality-red.csv'

dataset1 = np.loadtxt(file1, delimiter=';', skiprows=1)
names1 = np.genfromtxt(file1, delimiter=';', dtype=str, max_rows=1)

rate_learn = 1e-4
rate_forget = 1e-4  # SG.rate_forgetting_len(dataset1)
precision_w = 5e-2
precision_q = 3

options1 = {
	'sample': SG.Sample(dataset1),
	'learning_rate': rate_learn,
	'forgetting_rate': rate_forget,
	'weights_precision': precision_w,
	'quality_precision': precision_q
}

options1_diff = dict(options1)
options1_diff['sample'] = SG.Sample(dataset1, add_const_attr=True)

sg1_diff = SGDL(**options1_diff)
print('Calculate:', sg1_diff.calculate())
sg1_diff.info()

# sg1_vect = SGVL(**options1)
# print('Calculate:', sg1_vect.calculate())
# sg1_vect.info()

import matplotlib.pyplot as plt

x = np.array([precedent.x for precedent in options1['sample']])
y = np.array([precedent.y for precedent in options1['sample']])

sg2y = np.array([sg1_diff.algorithm(sg1_diff.weights, precedent.x) for precedent in options1['sample']])

# plt.plot(y, 'r*', sg2y, 'b.', alpha=0.1)
plt.plot(sg2y, 'b.', alpha=0.1)

# plt.show()
