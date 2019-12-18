import plotly.graph_objects as go
import numpy as np


def fig_get_new():
	return go.Figure()


def dataset_scatter(dataset, names):
	fig = go.Figure()
	dataset = dataset.transpose()
	for i in range(len(names)):
		fig.add_trace(go.Scatter(
			y=dataset[i],
			name=names[i],
			mode='markers'
		))
	fig.show()


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
			name=self.name_dataset
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
		self.histogram()
		self.scatter1()
		self.scatter2()
