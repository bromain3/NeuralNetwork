import numpy as np

"""
Add error handling, input size must match weights, type must be np array
Add print handling so it looks nice
"""

class Neuron:

    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights

    def output(self, input):
        # no np dot usage : return(np.sum(input * self.weights) + self.bias)
        # dot prod version:
        return(np.dot(input, self.weights) + self.bias)






if __name__ == '__main__':
    bias = 2
    weights = [0.2, 0.8, -0.5]
    input = [1, 2, 3]

    myneuron = Neuron(
        bias=bias,
        weights=np.array(weights)
    )
    print("hand calculated output :", input[0]*weights[0]+input[1]*weights[1]+input[2]*weights[2]+bias)
    print("Neuron calculated output :", myneuron.output(np.array(input)))