import stochastic_gradient as SG
import plotly.graph_objects as go
import numpy as np


class SGL(SG.Default):
	def __init__(self, sample, learning_rate, forgetting_rate, quality_precision, weights_precision):
		super().__init__(sample, learning_rate, forgetting_rate, quality_precision, weights_precision)
		self.log_w, self.log_q, self.log_p, self.log_wd, self.log_qd, self.log_qs = [], [], [], [], [], []
		self.log_cond = []

	def _set_weights(self, new_weights):
		self.log_w.append(list(self.weights))
		super()._set_weights(new_weights)

	def _set_quality(self, new_quality):
		self.log_q.append(str(self.quality_) + ' (' + str(self.quality()) + ')')
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
		""" May overload console buffer! """
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


class SGLDeriv(SGL, SG.Deriv):
	pass


class SGLVect(SGL, SG.Vect):
	pass


class Graphs:
	def __init__(self, sg, dataset_name, algorithm_name):
		self.name_dataset = dataset_name
		self.name_algorithm = algorithm_name
		self.y = np.array([precedent.y for precedent in sg.sample])
		self.ya = np.array([sg.algorithm(sg.weights, precedent.x) for precedent in sg.sample])

	def histogram_init(self, fig):
		fig.add_trace(go.Histogram(
			x=self.y,
			name=self.name_dataset
		))

	def histogram_add(self, fig, ya=None, name=None):
		if ya is None:
			ya = self.ya
		if name is None:
			name = self.name_algorithm
		fig.add_trace(go.Histogram(
			x=ya,
			name=name
		))

	def histogram(self):
		fig = go.Figure()
		self.histogram_init(fig)
		self.histogram_add(fig)
		fig.show()

	def scatter1_init(self, fig):
		fig.add_trace(go.Scatter(
			y=self.y,
			name=self.name_dataset,
			mode='markers'
		))

	def scatter1_add(self, fig, ya=None, name=None, size=5, opacity=0.5):
		if ya is None:
			ya = self.ya
		if name is None:
			name = self.name_algorithm
		fig.add_trace(go.Scatter(
			y=ya,
			name=name,
			mode='markers',
			marker=dict(
				size=size,
				opacity=opacity
			),
			error_y=dict(
				type='data',
				array=self.y - ya,
				thickness=0.3,
				width=2,
				symmetric=False
			)
		))

	def scatter1(self):
		fig = go.Figure()
		self.scatter1_init(fig)
		self.scatter1_add(fig)
		fig.show()

	def scatter2_init(self, fig):
		fig.add_trace(go.Scatter(
			x=self.y,
			y=self.y,
			name=self.name_dataset,
		))
		fig.update_layout(
			xaxis_title_text=self.name_dataset,
			yaxis_title_text=self.name_algorithm
		)

	def scatter2_add(self, fig, ya=None, name=None, size=10, opacity=0.1):
		if ya is None:
			ya = self.ya
		if name is None:
			name = self.name_algorithm
		fig.add_trace(go.Scatter(
			x=self.y,
			y=ya,
			name=name,
			mode='markers',
			marker=dict(
				size=size,
				opacity=opacity
			)
		))

	def scatter2(self):
		fig = go.Figure()
		self.scatter2_init(fig)
		self.scatter2_add(fig)
		fig.show()

	def all(self):
		self.scatter1()
		self.histogram()
		self.scatter2()


def fig_get_new():
	return go.Figure()
