import matplotlib.pyplot as plt
import NeuronLayer as nl
import ActivationFunctions as af
import numpy as np
import LossFunctions as lf
import Optimizer as opt
import random

def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return (X, y)

accuracy = lf.Accuracy()
X, y = create_data(100, 3)
#X = np.linspace(-5,5,100000)
#y = np.sin(X)
#shuffled_indexe = [i for i in range(90000)]
#random.shuffle(shuffled_indexe)
#X_shuf = X[shuffled_indexe]
#y_shuf = np.where(y[shuffled_indexe] > X_shuf, 1,0)
#X_shuf = X_shuf.reshape((len(shuffled_indexe), 1))
#print(y_shuf)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
#plt.plot(X,y)
#plt.show()

dense1 = nl.NeuronLayer(2, 64)
activation1 = af.ReLU()
dense2 = nl.NeuronLayer(64, 3)
activation2 = lf.CombinedSoftmaxCrossEntropy()
optimizer = opt.AdaptativeMomentum(learningRate=0.02, decay=1e-5)

L = []
for i in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = activation2.forward(dense2.output, y)
    acc = accuracy.calculate(activation2.output, y)
    #print(acsm2.output, loss)

    activation2.backward(activation2.output, y)
    dense2.backward(activation2.inputGradient)
    activation1.backward(dense2.inputGradient)
    dense1.backward(activation1.inputGradient)
    #print(smLayer.inputGradient)

    optimizer.preUpdateParams()
    optimizer.updateParams(dense1)
    optimizer.updateParams(dense2)
    optimizer.postUpdateParams()

    if i%100 == 0:
        print('loss:', loss)
        print('acc:', acc)
        print('epoch:', i)
        print('learning rate:', optimizer.currentLearningRate)
        L.append(loss)


plt.plot(L)
plt.show()