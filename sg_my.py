import stochastic_gradient as SG


class SGL(SG.Default):
	def __init__(self, sample, learning_rate, forgetting_rate, quality_precision, weights_precision):
		super().__init__(sample, learning_rate, forgetting_rate, quality_precision, weights_precision)
		self.log_w, self.log_q, self.log_p, self.log_wd, self.log_qd, self.log_qs = [], [], [], [], [], []
		self.log_cond = []

	def _set_weights(self, new_weights):
		self.log_w.append(list(self.weights))
		super()._set_weights(new_weights)

	def _set_quality(self, new_quality):
		self.log_q.append(str(self.quality) + ' (' + str(self.q()) + ' diff:' + str(self.quality - self.q()) + ')')
		super()._set_quality(new_quality)

	def _calc_step(self, index):
		result = super()._calc_step(index)
		self.log_p.append(result)
		self.log_qd.append(self.quality_diff())
		self.log_qs.append(self.stability_quality())
		self.log_wd.append(self.weights_diff())
		if self.is_stable_quality():
			cond = 'q'
		elif self.is_stable_weights():
			cond = 'w'
		else:
			cond = ''
		self.log_cond.append(cond)
		return result

	def info_log(self):
		""" May overload console buffer! """
		print('log_p:', self.log_p)
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


class SGLActiv(SGL, SG.Activ):
	pass


class SGLSimple(SGL, SG.Simple):
	pass
