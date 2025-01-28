import numpy as np
import ActivationFunctions

"""
Use abstract methodes
"""

class AbstractLoss:

    def __init__(self):
        self.inputGradient = None

    def calculate(self, prediction, target):
        sampleLoss = self.forward(prediction, target)
        dataLoss = np.mean(sampleLoss)
        return(dataLoss)

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

class Accuracy:
    def calculate(self, prediction, target):
        predictions = np.argmax(prediction, axis=1)
        if len(target.shape) == 2:
            target= np.argmax(target, axis=1)
        accuracy = np.mean(predictions == target)
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
    a = Accuracy()
    print(a.calculate(dummy_output, target_out))