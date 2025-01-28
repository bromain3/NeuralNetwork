import numpy  as np

"""
add error handlingand input size handling
define abstract class
"""


"""Sigmoid not tested"""
class Sigmoid:
    def __init__(self):
        self.output = None
        self.input = None
        self.inputGradient = None
        self.inputGradient = None

    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))

    def backward(self, incomingGradients):
        self.inputGradient = incomingGradients * (1 - self.output) * self.output

class Softmax:
    def __init__(self):
        self.output = None
        self.input = None
        self.inputGradient = None

    def forward(self, vector):
        """
        Only works for 2D array (ie batch input)
        """
        self.input = vector
        e = np.exp(vector - np.max(vector, axis=1, keepdims=True))
        self.output = e / np.sum(e, axis=1, keepdims=True)

    def backward(self, incomingGradients):
        self.inputGradient = np.empty_like(incomingGradients)
        for index, (singleOutput,singleIncomingGradient) in enumerate(zip(self.output, incomingGradients)):
            singleOutput= singleOutput.reshape(-1, 1)
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput,singleOutput.T)
            self.inputGradient[index] = np.dot(jacobianMatrix, singleIncomingGradient)

class Linear:
    def __init__(self):
        self.output = None
        self.input = None
        self.inputGradient = None

    def forward(self, vector):
        self.input = vector
        self.output = vector

    def backward(self, incomingGradients):
        self.inputGradient = incomingGradients.copy()



class ReLU:
    def __init__(self):
        self.output = None
        self.input = None
        self.inputGradient = None

    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)

    def backward(self, incomingGradients):
        self.inputGradient = incomingGradients.copy()
        self.inputGradient[self.input <= 0] = 0



if __name__ =='__main__':
    import matplotlib.pyplot as plt
    range = np.linspace(-5, 5, 1000)
    myReLu = ReLU()
    myReLu.forward(range)
    plt.plot(range, myReLu.output)
    plt.show()

    sm = Softmax()
    sm.forward(np.array([[4.8,1.21,2.385]]))
    print(sm.output)