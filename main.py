import numpy as np

import stochastic_gradient as SG
import sg_my
import sg_graphs as g


def set_random_seed(new_seed=None):
	if new_seed is None:
		return
	np.random.seed(new_seed)


random_seed = 0
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
file1 = 'samples/Wine Quality Data Set/winequality-red.csv'

dataset1 = np.loadtxt(file1, delimiter=';', skiprows=1)
dataset1_name = 'Red Wine Quality'
dataset1_names = np.genfromtxt(file1, delimiter=';', dtype=str, max_rows=1)
# xy = {name: x_i for name, x_i in zip(dataset1_names, dataset1.transpose())}
# g.dataset_scatter(dataset1, dataset1_names)


# 1.1
class MyLA_round(sg_my.SGLActiv):
	def algorithm(self, weights, x):
		alg = super().algorithm(weights, x)
		return round(alg)


set_random_seed(random_seed)
options1_a = dict(
	sample=SG.Sample(dataset1, add_const_attr=True),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=5e-5,
	weights_precision=0
)

sgla1 = MyLA_round(**options1_a)
sgla1_name = 'SG Activ'
print('Calculate', dataset1_name, sgla1_name, ':\n', sgla1.calculate())
sgla1.info()
# sgla1.info_log()
sgla1_g = g.Graphs(sgla1, dataset1_name, sgla1_name)
# sgla1_g.all()


# 1.2
set_random_seed(random_seed)
options1_a2 = dict(
	sample=SG.Sample(dataset1, add_const_attr=True),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=9e-6,
	weights_precision=5e-5
)

sgla1_2 = sg_my.SGLActiv(**options1_a2)
sgla1_2_name = 'SG Activ float'
print('Calculate', dataset1_name, sgla1_2_name, ':\n', sgla1_2.calculate())
sgla1_2.info()
# sgla1_2.info_log()
sgla1_2_g = g.Graphs(sgla1_2, dataset1_name, sgla1_2_name)
# sgla1_2_g.all()


# 1.3
class MyLS_round(sg_my.SGLSimple):
	def algorithm(self, weights, x):
		alg = super().algorithm(weights, x)
		return round(alg)


set_random_seed(random_seed)
options1_s = dict(
	sample=SG.Sample(dataset1),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=5e-5,
	weights_precision=0
)

sgls1 = MyLS_round(**options1_s)
sgls1_name = 'SG Simple'
print('Calculate', dataset1_name, sgls1_name, ':\n', sgls1.calculate())
sgls1.info()
# sgls1.info_log()
sgls1_g = g.Graphs(sgls1, dataset1_name, sgls1_name)
# sgls1_g.all()


fig1 = g.fig_get_new()
sgla1_g.scatter1_init(fig1)
sgla1_g.scatter1_add(fig1)
sgla1_g.scatter1_add(fig1, sgla1_2_g.ya.round(), sgla1_2_g.name_algorithm)
sgla1_g.scatter1_add(fig1, sgls1_g.ya, sgls1_g.name_algorithm)
# fig1.show()

fig2 = g.fig_get_new()
sgla1_g.histogram_init(fig2)
sgla1_g.histogram_add(fig2)
sgla1_g.histogram_add(fig2, sgla1_2_g.ya.round(), sgla1_2_g.name_algorithm)
sgla1_g.histogram_add(fig2, sgls1_g.ya, sgls1_g.name_algorithm)
# fig2.show()

fig3 = g.fig_get_new()
sgla1_g.scatter2_init(fig3)
sgla1_g.scatter2_add(fig3)
sgla1_g.scatter2_add(fig3, sgla1_2_g.ya.round(), sgla1_2_g.name_algorithm)
sgla1_g.scatter2_add(fig3, sgls1_g.ya, sgls1_g.name_algorithm)
# fig3.show()


file2 = 'samples/Wine Quality Data Set/winequality-white.csv'

dataset2 = np.loadtxt(file2, delimiter=';', skiprows=1)
dataset2_name = 'White Wine Quality'
dataset2_names = np.genfromtxt(file2, delimiter=';', dtype=str, max_rows=1)
# g.dataset_scatter(dataset2, dataset2_names)

set_random_seed(random_seed)
options2_s = dict(
	sample=SG.Sample(dataset2),
	learning_rate=1e-7,
	forgetting_rate=1e-4,
	quality_precision=5e-5,
	weights_precision=0
)

sgls2 = MyLS_round(**options2_s)
sgls2_name = sgls1_name
print('Calculate', dataset2_name, sgls2_name, ':\n', sgls2.calculate())
sgls2.info()
# sgls2.info_log()
sgls2_g = g.Graphs(sgls1, dataset2_name, sgls2_name)
# sgls2_g.all()


# https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat
file3 = 'samples/airfoil_self_noise.dat'
dataset3 = np.loadtxt(file3)
dataset3_name = 'Airfoil Self-Noise'
dataset3_names = ['Frequency, Hz', 'Angle of attack, degr', 'Chord length, m', 'Free-stream velocity, m/s',
				  'Suction side displacement thickness, m', 'Scaled sound pressure level, db']
# g.dataset_scatter(dataset3, dataset3_names)

set_random_seed(random_seed)
options3_s = dict(
	sample=SG.Sample(dataset3),
	learning_rate=1e-12,
	forgetting_rate=1e-6,
	quality_precision=5e-5,
	weights_precision=1e-5
)

sgls3 = sg_my.SGLSimple(**options3_s)
sgls3_name = sgls1_name
print('Calculate', dataset3_name, sgls3_name, ':\n', sgls3.calculate())
sgls3.info()
sgls3.info_log()
sgls3_g = g.Graphs(sgls3, dataset3_name, sgls3_name)
sgls3_g.all()
