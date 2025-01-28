import numpy as np
import ActivationFunctions

"""
Use abstract methodes
classer les différentes classes dans différents fichiers
"""

class AbstractLoss:

    def __init__(self):
        self.inputGradient = None

    def calculate(self, prediction, target):
        sampleLoss = self.forward(prediction, target)
        dataLoss = np.mean(sampleLoss)
        return(dataLoss)


"""LossBinaryCrossEntropy not tested"""
class LossBinaryCrossEntropy(AbstractLoss):

    def forward(self, prediction, target):
        clippedPredictions = np.clip(prediction, 1e-7, 1 - 1e-7)
        sampleLoss = - (target * np.log(clippedPredictions) + (1 - target) * np.log(1 - clippedPredictions))
        sampleLoss = np.mean(sampleLoss, axis=-1)
        return(sampleLoss)

    def backward(self, prediction, target):
        samples = len(prediction)
        output = len(prediction[0])
        clippedPredictions = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.inputGradient = - (target / clippedPredictions - (1 - target) / (1 - clippedPredictions)) / prediction / samples

class LossCategoricalCrossEntropy(AbstractLoss):

    def forward(self, prediction, target):
        samples = len(prediction)
        clippedPredictions = np.clip(prediction, 1e-7, 1 - 1e-7)
        if len(target.shape) == 1:
            correct_confidences = clippedPredictions[range(samples), target]
        elif len(target.shape) == 2:
            print(clippedPredictions, target)
            correct_confidences = np.sum(clippedPredictions * target, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return(negative_log_likelihoods)

    def backward(self, prediction, target):
        samples = len(prediction)
        labels = len(prediction[0])
        if len(target.shape) == 1:
            target = np.eye(labels)[target]
        self.inputGradient = - target / prediction / samples

class LossMeanSquaredError(AbstractLoss):

    def forward(self, prediction, target):
        sampleLoss = np.mean((target - prediction) ** 2, axis=-1)
        return(sampleLoss)

    def backward(self, prediction, target):
        samples = len(prediction)
        output = len(prediction[0])
        self.inputGradient = - 2 * (target - prediction) / output / samples

"""LossMeanAbsoluteError seems not to work"""
class LossMeanAbsoluteError(AbstractLoss):

    def forward(self, prediction, target):
        sampleLoss = np.mean(np.abs(target - prediction), axis=-1)
        return(sampleLoss)

    def backward(self, prediction, target):
        samples = len(prediction)
        output = len(prediction[0])
        self.inputGradient = np.sign(target - prediction) / output / samples

class combinedLinearMeanSquaredError:

    def __init__(self):
        self.activation = ActivationFunctions.Linear()
        self.loss = LossMeanSquaredError()
        self.output = None
        self.inputGradient = None

    def forward(self, inputs, target):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return(self.loss.calculate(self.output, target))

    def backward(self, prediction, target):
        self.loss.backward(prediction, target)
        self.activation.backward(self.loss.inputGradient)
        self.inputGradient = self.activation.inputGradient

"""combinedLinearMeanAbsoluteError seems not to work"""
class combinedLinearMeanAbsoluteError:

    def __init__(self):
        self.activation = ActivationFunctions.Linear()
        self.loss = LossMeanAbsoluteError()
        self.output = None
        self.inputGradient = None

    def forward(self, inputs, target):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return(self.loss.calculate(self.output, target))

    def backward(self, prediction, target):
        self.loss.backward(prediction, target)
        self.activation.backward(self.loss.inputGradient)
        self.inputGradient = self.activation.inputGradient


class CombinedSoftmaxCrossEntropy:

    def __init__(self):
        self.activation = ActivationFunctions.Softmax()
        self.loss = LossCategoricalCrossEntropy()
        self.output = None
        self.inputGradient = None

    def forward(self, inputs, target):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return(self.loss.calculate(self.output, target))

    def backward(self, prediction, target):
        samples = len(prediction)
        if len(target.shape) == 2:
            target = np.argmax(target, axis=1)
        self.inputGradient = prediction.copy()
        self.inputGradient[range(samples), target] -= 1
        self.inputGradient = self.inputGradient / samples


"""combinedSigmoidBinaryCrossEntropy not tested"""
class combinedSigmoidBinaryCrossEntropy:

    def __init__(self):
        self.activation = ActivationFunctions.Sigmoid()
        self.loss = LossBinaryCrossEntropy()
        self.output = None
        self.inputGradient = None

    def forward(self, inputs, target):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return(self.loss.calculate(self.output, target))

    def backward(self, prediction, target):
        self.loss.backward(prediction, target)
        self.activation.backward(self.loss.inputGradient)
        self.inputGradient = self.activation.inputGradient


class SoftmaxClassificationAccuracy:
    def calculate(self, prediction, target):
        predictions = np.argmax(prediction, axis=1)
        if len(target.shape) == 2:
            target= np.argmax(target, axis=1)
        accuracy = np.mean(predictions == target)
        return(accuracy)

class LinearRegressionAccuracy:
    def __init__(self, tol= 1/250):
        self.tol = tol

    def calculate(self, prediction, target):
        accuracy = np.mean(np.absolute(prediction - target) < np.std(target) * self.tol)
        return(accuracy)

"""SigmoidBinaryClassificationAccuracy not tested"""
class SigmoidBinaryClassificationAccuracy:
    def calculate(self, prediction, target):
        pred = (prediction > 0.5) * 1
        accuracy = np.mean(pred == target)
        return(accuracy)

if __name__ =='__main__':
    dummy_output = np.array([[0.7,0.1,0.2]])
    target_out = np.array([[1,0,0]])
    a = LossCategoricalCrossEntropy()
    print(a.calculate(dummy_output,target_out))

    dummy_output = np.array([[0.7, 0.2, 0.1],  # defines confidence for each class
                             [0.5, 0.1, 0.4],
                             [0.02, 0.9, 0.08]])
    target_out = np.array([1, 0, 0]) # defines wich class is the right one
    a = SoftmaxClassificationAccuracy()
    print(a.calculate(dummy_output, target_out))