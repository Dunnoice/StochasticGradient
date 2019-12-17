import numpy as np

import stochastic_gradient as SG
import sg_my


def set_random_seed(new_seed=None):
	if new_seed is None:
		return
	np.random.seed(new_seed)


random_seed = 0
file1 = 'samples/Wine Quality Data Set/winequality-red.csv'

dataset1 = np.loadtxt(file1, delimiter=';', skiprows=1)
dataset1_names = np.genfromtxt(file1, delimiter=';', dtype=str, max_rows=1)
# xy = {name: x_i for name, x_i in zip(dataset1_names, dataset1.transpose())}
dataset1_name = 'Red Wine Quality'
dataset1_ypos = len(dataset1[0]) - 1


class MyDerivLround(sg_my.SGLDeriv):
	def algorithm(self, weights, x):
		alg = super().algorithm(weights, x)
		return round(alg)


set_random_seed(random_seed)
options1_diff = dict(
	sample=SG.Sample(dataset1, dataset1_ypos, add_const_attr=True),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=5e-5,
	weights_precision=0,
)

sgld1 = MyDerivLround(**options1_diff)
sgld1_name = 'SGderiv'
print('Calculate', sgld1_name, ':\n', sgld1.calculate())
sgld1.info()
# sgld1.info_log()
sgld1_g = sg_my.Graphs(sgld1, dataset1_name, sgld1_name)
# sgld1_g.all()


set_random_seed(random_seed)
options1_diff2 = dict(
	sample=SG.Sample(dataset1, dataset1_ypos, add_const_attr=True),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=9e-6,
	weights_precision=5e-5,
)

sgld1_2 = sg_my.SGLDeriv(**options1_diff2)
sgld1_2_name = 'SGderiv float'
print('Calculate', sgld1_2_name, ':\n', sgld1_2.calculate())
sgld1_2.info()
# sgld1_2.info_log()
sgld1_2_g = sg_my.Graphs(sgld1_2, dataset1_name, sgld1_2_name)
# sgld1_2_g.all()


set_random_seed(random_seed)
options1_vect = dict(
	sample=SG.Sample(dataset1, dataset1_ypos),
	learning_rate=1e-5,
	forgetting_rate=1e-4,
	quality_precision=5e-5,
	weights_precision=0,
)


class MyVectLround(sg_my.SGLVect):
	def algorithm(self, weights, x):
		alg = super().algorithm(weights, x)
		return round(alg)


sglv1 = MyVectLround(**options1_vect)
sglv1_name = 'SGvect'
print('Calculate', sglv1_name, ':\n', sglv1.calculate())
sglv1.info()
# sglv1.info_log()
sglv1_g = sg_my.Graphs(sglv1, dataset1_name, sglv1_name)
# sglv1_g.all()


class MyVectLround2(MyVectLround):  # TODO test
	def loss(self, y1, y2=1):
		return SG.loss_binary_approx(y1, y2)

	def loss_deriv(self, y1, y2=1):
		return y2


fig1 = sg_my.fig_get_new()
sgld1_g.scatter1_init(fig1)
sgld1_g.scatter1_add(fig1)
sgld1_g.scatter1_add(fig1, sgld1_2_g.ya.round(), sgld1_2_g.name_algorithm)
sgld1_g.scatter1_add(fig1, sglv1_g.ya, sglv1_g.name_algorithm)
# fig1.show()

fig2 = sg_my.fig_get_new()
sgld1_g.histogram_init(fig2)
sgld1_g.histogram_add(fig2)
sgld1_g.histogram_add(fig2, sgld1_2_g.ya.round(), sgld1_2_g.name_algorithm)
sgld1_g.histogram_add(fig2, sglv1_g.ya, sglv1_g.name_algorithm)
# fig2.show()

fig3 = sg_my.fig_get_new()
sgld1_g.scatter2_init(fig3)
sgld1_g.scatter2_add(fig3)
sgld1_g.scatter2_add(fig3, sgld1_2_g.ya.round(), sgld1_2_g.name_algorithm)
sgld1_g.scatter2_add(fig3, sglv1_g.ya, sglv1_g.name_algorithm)
# fig3.show()


file2 = 'samples/Wine Quality Data Set/winequality-white.csv'

dataset2 = np.loadtxt(file2, delimiter=';', skiprows=1)
dataset2_names = np.genfromtxt(file2, delimiter=';', dtype=str, max_rows=1)
dataset2_name = 'White Wine Quality'
dataset2_ypos = len(dataset2[0]) - 1

set_random_seed(random_seed)
options2_diff = dict(
	sample=SG.Sample(dataset2, dataset2_ypos, add_const_attr=True),
	learning_rate=1e-5,
	forgetting_rate=SG.rate_forgetting_len(len(dataset2)),
	quality_precision=5e-5,
	weights_precision=0,
)

sgld2 = MyDerivLround(**options2_diff)
sgld2_name = 'SGderiv'
print('Calculate', sgld2_name, ':\n', sgld2.calculate())
sgld2.info()
sgld2.info_log()
sgld2_g = sg_my.Graphs(sglv1, dataset2_name, sgld2_name)
sgld2_g.all()
