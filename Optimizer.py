import numpy as np

"""
can be cleaned by spliting between multiple classes and pretratment for if statments
add abstract methode
"""


class StochasticGradientDescent:

    def __init__(self, learningRate=1.0, decay=0.0, momentum=0.0):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1 / (1 + self.decay * self.iterations))

    def updateParams(self, neuronLayer):

        if self.momentum:
            if not hasattr(neuronLayer, 'weightMomentum'):
                neuronLayer.weightMomentum = np.zeros_like(neuronLayer.weights)
                neuronLayer.biasMomentum = np.zeros_like(neuronLayer.biases)
            weightUpdate = self.momentum * neuronLayer.weightMomentum - self.currentLearningRate * neuronLayer.weightsGradient
            biasUpdate = self.momentum * neuronLayer.biasMomentum - self.currentLearningRate * neuronLayer.biasGradient
            neuronLayer.weightMomentum = weightUpdate
            neuronLayer.biasMomentum = biasUpdate

        else:
            weightUpdate = - self.currentLearningRate * neuronLayer.weightsGradient
            biasUpdate = - self.currentLearningRate * neuronLayer.biasGradient

        neuronLayer.weights += weightUpdate
        neuronLayer.biases += biasUpdate

    def postUpdateParams(self):
        self.iterations += 1

class AdaptativeGradient:

    def __init__(self, learningRate=1.0, decay=0.0, epsilon=1e-7):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1 / (1 + self.decay * self.iterations))

    def updateParams(self, neuronLayer):

        if not hasattr(neuronLayer, 'weightCache'):
            neuronLayer.weightCache = np.zeros_like(neuronLayer.weights)
            neuronLayer.biasCache = np.zeros_like(neuronLayer.biases)

        neuronLayer.weightCache += neuronLayer.weightsGradient ** 2
        neuronLayer.biasCache += neuronLayer.biasGradient ** 2

        neuronLayer.weights += - self.currentLearningRate * neuronLayer.weightsGradient / (np.sqrt(neuronLayer.weightCache) + self.epsilon)
        neuronLayer.biases += - self.currentLearningRate * neuronLayer.biasGradient / (np.sqrt(neuronLayer.biasCache) + self.epsilon)


    def postUpdateParams(self):
        self.iterations += 1



class RMSPropagation:

    def __init__(self, learningRate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0

    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1 / (1 + self.decay * self.iterations))

    def updateParams(self, neuronLayer):

        if not hasattr(neuronLayer, 'weightCache'):
            neuronLayer.weightCache = np.zeros_like(neuronLayer.weights)
            neuronLayer.biasCache = np.zeros_like(neuronLayer.biases)

        neuronLayer.weightCache = self.rho * neuronLayer.weightCache + (1 - self.rho) * neuronLayer.weightsGradient ** 2
        neuronLayer.biasCache = self.rho * neuronLayer.biasCache + (1 - self.rho) * neuronLayer.biasGradient ** 2

        neuronLayer.weights += - self.currentLearningRate * neuronLayer.weightsGradient / (np.sqrt(neuronLayer.weightCache) + self.epsilon)
        neuronLayer.biases += - self.currentLearningRate * neuronLayer.biasGradient / (np.sqrt(neuronLayer.biasCache) + self.epsilon)


    def postUpdateParams(self):
        self.iterations += 1