import plotly.graph_objects as go
from main import *

x1 = np.array([precedent.x for precedent in options1_diff['sample']])
y1 = np.array([precedent.y for precedent in options1_diff['sample']])
xy1 = {name: x_i for name, x_i in zip(names1, dataset1.transpose())}

sgd1_y = np.array([sgld1.algorithm(sgld1.weights, precedent.x) for precedent in options1_diff['sample']])
sgd1_y2 = np.array([sgld1_2.algorithm(sgld1_2.weights, precedent.x) for precedent in options1_diff2['sample']])
sgv1_y = np.array([sglv1.algorithm(sglv1.weights, precedent.x) for precedent in options1_vect['sample']])


def scatter_error(fig, y, y_new, opacity=1.0):
	fig.add_trace(go.Scatter(
		y=y,
		mode='markers',
		error_y=dict(
			type='data',
			array=y_new - y,
			thickness=0.3,
			width=2,
			symmetric=False,
		),
		marker=dict(
			opacity=opacity
		)
	))


fig1_y = go.Figure()
fig1_y.add_trace(go.Scatter(
	y=y1,
	name='dataset 1',
	mode='markers',
	marker=dict(
		opacity=1
	)
))
scatter_error(fig1_y, sgd1_y, y1)
scatter_error(fig1_y, sgd1_y2.round(), y1, 0.5)
scatter_error(fig1_y, sgv1_y, y1, 0.5)
fig1_y.show()


fig1_h = go.Figure()
fig1_h.add_trace(go.Histogram(
	x=y1,
	name='dataset 1',
	opacity=0.7,
))
fig1_h.add_trace(go.Histogram(
	x=sgd1_y,
	name='SGdiff',
	opacity=0.7,
))
fig1_h.add_trace(go.Histogram(
	x=sgd1_y2.round(),
	name='SGdiff 2',
	opacity=0.7,
))
fig1_h.add_trace(go.Histogram(
	x=sgv1_y,
	name='SGvect',
	opacity=0.7,
))
fig1_h.show()


fig1_all = go.Figure()
fig1_all.add_trace(go.Scatter(
	x=y1,
	y=y1,
	name='dataset 1',
))
fig1_all.add_trace(go.Scatter(
	x=y1,
	y=sgd1_y,
	name='SGdiff',
	mode='markers',
	marker=dict(
		opacity=0.1
	)
))
fig1_all.add_trace(go.Scatter(
	x=y1,
	y=sgd1_y2.round(),
	name='SGdiff 2',
	mode='markers',
	marker=dict(
		opacity=0.1
	)
))
fig1_all.add_trace(go.Scatter(
	x=y1,
	y=sgv1_y,
	name='SGvect',
	mode='markers',
	marker=dict(
		opacity=0.1
	)
))
fig1_all.update_layout(
	xaxis_title_text='original',
	yaxis_title_text='experimental'
)
fig1_all.show()
