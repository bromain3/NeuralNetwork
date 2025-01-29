import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import NeuralNetwork as nn
import ActivationFunctions as af
import LossFunctions as lf
import Optimizer as opt


gas1 = ct.Solution('gri30.yaml')

X = np.zeros((10000,3))
y = np.zeros((10000,1))

for i in range(10000):
    a = np.random.rand(3)
    a = a / np.sum(a)
    gas1.TPX = 300, 101325, {'CH4':a[0], 'O2':a[1], 'N2':a[2]}

    gas1.equilibrate('HP')


    X[i,:] = a
    y[i,:] = gas1.T


print(X,y)

neural = nn.NeuralNetwork(3, 1, [10,10])
neural.setActivationFunction(0, af.ReLU())
neural.setActivationFunction(1, af.ReLU())
neural.setActivationFunction(2, lf.CombinedLinearMeanSquaredError())
neural.setAcuracyFunction(lf.LinearRegressionAccuracy())
#neural.setOptimizerFunction(opt.AdaptativeMomentum(learningRate=0.005, decay=1e-3))
neural.setOptimizerFunction(opt.AdaptativeMomentum())


L, A, Lr = neural.train(X, y, 1000, batchSize=1000,bigBatchSize='all')
plt.plot(L)
plt.show()
plt.plot(A)
plt.show()