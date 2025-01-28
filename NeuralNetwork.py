import NeuronLayer as nl
import numpy as np

"""
Add error handling
découper encor plus forward et backward (rajouter une fonction par couche) pour la compréhension
add auto data shuffling
"""


class NeuralNetwork:
    """
    Dense neural network
    """

    def __init__(self, ninputs, noutputs, nbNeuronHiddenLayers):
        self.layers = []
        if len(nbNeuronHiddenLayers) == 0:
            self.layers.append(nl.NeuronLayer(ninputs, noutputs))

        elif len(nbNeuronHiddenLayers) == 1:
            self.layers.append(nl.NeuronLayer(ninputs, nbNeuronHiddenLayers[0]))
            self.layers.append(nl.NeuronLayer(nbNeuronHiddenLayers[0], noutputs))

        else:
            self.layers.append(nl.NeuronLayer(ninputs, nbNeuronHiddenLayers[0]))
            for curentNbNeuron,previousNbNeuron in zip(nbNeuronHiddenLayers[1:], nbNeuronHiddenLayers[0:-1]):
                self.layers.append(nl.NeuronLayer(previousNbNeuron, curentNbNeuron))
            self.layers.append(nl.NeuronLayer(nbNeuronHiddenLayers[-1], noutputs))
        self.activationFunctions = [None for _ in range(len(nbNeuronHiddenLayers)+1)]

    def setActivationFunction(self, layer, activationFunction):
        self.activationFunctions[layer] = activationFunction

    def setOptimizerFunction(self, optimizerObject):
        self.optimizer = optimizerObject

    def setAcuracyFunction(self, accuracyObject):
        self.accuracy = accuracyObject

    def forward(self, inputs, y):
        """
        # not tested
        if len(self.layers) == 1:
            self.layers[0].forward(inputs)
            loss = self.activationFunctions[0].forward(self.layers[0].output, y)
        """
        if len(self.layers) == 1:
            pass
        else:
            self.layers[0].forward(inputs)
            self.activationFunctions[0].forward(self.layers[0].output)
            for i in range(1,len(self.layers)-1):
                self.layers[i].forward(self.activationFunctions[i-1].output)
                self.activationFunctions[i].forward(self.layers[i].output)
            self.layers[-1].forward(self.activationFunctions[-2].output)
            loss = self.activationFunctions[-1].forward(self.layers[-1].output, y)
        accuracy = self.accuracy.calculate(self.activationFunctions[-1].output, y)
        return(self.activationFunctions[-1].output, loss, accuracy)

    def predict(self, inputs):
        """
        # not tested
        if len(self.layers) == 1:

        """
        if len(self.layers) == 1:
            pass
        else:
            self.layers[0].forward(inputs)
            self.activationFunctions[0].forward(self.layers[0].output)
            for i in range(1, len(self.layers) - 1):
                self.layers[i].forward(self.activationFunctions[i - 1].output)
                self.activationFunctions[i].forward(self.layers[i].output)
            self.layers[-1].forward(self.activationFunctions[-2].output)
            self.activationFunctions[-1].forward(self.layers[-1].output, np.zeros_like(self.layers[-1].output))
        return (self.activationFunctions[-1].output)

    def backward(self, output, y):
        """
        # not tested
        if len(self.layers) == 1:
            self.activationFunctions[0].backward(output, y)
            self.layers[0].backward(self.activationFunctions[0].inputGradient)
        """
        if len(self.layers) == 1:
            pass
        else:
            self.activationFunctions[-1].backward(output, y)
            self.layers[-1].backward(self.activationFunctions[-1].inputGradient)
            for i in range(1, len(self.layers)):
                backwardIndex = (len(self.layers) - 1) - i
                self.activationFunctions[backwardIndex].backward(self.layers[backwardIndex+1].inputGradient)
                self.layers[backwardIndex].backward(self.activationFunctions[backwardIndex].inputGradient)

    def optimize(self):
        self.optimizer.preUpdateParams()
        for layer in self.layers:
            self.optimizer.updateParams(layer)
        self.optimizer.postUpdateParams()

    def train(self, X, y, epoch, batchSize=128, bigBatchSize=None):
        ###### add data shuffling
        Loss = []
        Accuracy = []
        LearningRate = []
        steps = X.shape[0] // batchSize
        if steps * batchSize < X.shape[0]:
            steps += 1

        for i in range(epoch):
            for step in range(steps):
                batchX = X[step * batchSize:(step + 1) * batchSize]
                batchy = y[step * batchSize:(step + 1) * batchSize]

                result, loss, accuracy = self.forward(batchX, batchy)
                self.backward(result, batchy)
                self.optimize()
                if step % 100 == 0:
                    print('loss:', loss)
                    print('acc:', accuracy)
                    print('epoch:', i)
                    print('step:', step)
                    print('learning rate:', self.optimizer.currentLearningRate)
                    Loss.append(loss)
                    Accuracy.append(accuracy)
                    LearningRate.append(self.optimizer.currentLearningRate)

            # add this to epoch count ?
            if bigBatchSize == 'all':
                # Empirically adds great precision for low computation cost if you give all the data
                result, loss, accuracy = self.forward(X, y)
                self.backward(result, y)
                self.optimize()
            elif bigBatchSize is not None:
                keys = np.array(range(X.shape[0]))
                np.random.shuffle(keys)
                Xreduced = X[keys]
                yreduced = y[keys]
                result, loss, accuracy = self.forward(Xreduced, yreduced)
                self.backward(result, yreduced)
                self.optimize()

        return(Loss, Accuracy, LearningRate)


    def testNetwork(self, X, y):
        result, loss, accuracy = self.forward(X, y)
        return(result, loss, accuracy)


if __name__ == "__main__":
    import ActivationFunctions as af
    import LossFunctions as lf
    import Optimizer as opt

    nn = NeuralNetwork(2, 2, [10, 5])
    print(nn.layers)
    nn.setActivationFunction(0, af.ReLU())
    nn.setActivationFunction(1, af.ReLU())
    nn.setActivationFunction(2, lf.CombinedSoftmaxCrossEntropy())

    nn.setAcuracyFunction(lf.SoftmaxClassificationAccuracy())

    nn.setOptimizerFunction(opt.AdaptativeMomentum())
