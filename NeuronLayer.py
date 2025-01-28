import Neuron
import numpy as np

"""
add error handlingand input size handling
"""
np.random.seed(0)
class NeuronLayer:
    """
    Dense neuron layer
    """
    def __init__(self, ninputs, nneurons):
        self.biases = np.zeros((1,nneurons))
        self.weights = 0.1 * np.random.randn(ninputs, nneurons)
        self.output = None
        self.input = None
        self.weightsGradient = None
        self.biasGradient = None
        self.inputGradient = None

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, incommingGradients):
        self.weightsGradient = np.dot(self.input.T, incommingGradients)
        self.biasGradient = np.sum(incommingGradients, axis=0, keepdims=True)
        self.inputGradient = np.dot(incommingGradients, self.weights.T)

class NeuronLayer_eq_tst:
    def __init__(self, biases, weights):
        # np dot version
        self.biases = np.array(biases)
        self.weights = np.array(weights)
        # no np dot:
        #self.neurons = []
        #for b, w in zip(biases, weights):
        #    self.neurons.append(Neuron.Neuron(bias=b,weights=w))

    def output(self, inputs):
        # no np dot :
        #output = []
        #for n in self.neurons:
        #    output.append(n.output(input))
        #return(output)
        # dot prod version :
        return(np.dot(inputs, self.weights.T) + self.biases)






if __name__ =='__main__':
    import ActivationFunctions
    #### 3 neurones qui prenent 4 entrées (donc 3 sorties)
    bias1 = 2
    bias2 = 3
    bias3 = 0.5
    biases = [bias1,bias2,bias3]

    weights1 = [0.2, 0.8, -0.5, 1]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]
    weights = [weights1,weights2,weights3]

    mylayer1 = NeuronLayer_eq_tst(biases, weights)

    # 3 neurones qui prennet 3 entrées (donc 3 sorties
    bias1 = -1
    bias2 = 2
    bias3 = -0.5
    biases = [bias1, bias2, bias3]

    weights1 = [0.1, -0.14, 0.5]
    weights2 = [-0.5, 0.12, -0.33]
    weights3 = [-0.44, 0.73, -0.13]
    weights = [weights1, weights2, weights3]

    mylayer2 = NeuronLayer_eq_tst(biases, weights)

    input1 = [1, 2, 3, 2.5]
    input2 = [2, 5, -1, 2]
    input3 = [-1.5, 2.7, 3.3, -0.8]
    inputs = [input1, input2, input3]
    print("Layer1 output :", mylayer1.output(inputs))
    print("Layer2 output :", mylayer2.output(mylayer1.output(inputs)))

    layer1 = NeuronLayer(4,5)
    layer2 = NeuronLayer(5,2)
    layer1.forward(inputs)
    print(layer1.output)
    layer2.forward(layer1.output)
    print(layer2.output)

    def create_data(points, classes):
        X = np.zeros((points * classes, 2))
        y = np.zeros(points * classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(points * class_number, points * (class_number + 1))
            r = np.linspace(0, 1, points)
            t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
            X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
            y[ix] = class_number
        return(X,y)
    import matplotlib.pyplot as plt
    X,y = create_data(100,3)
    plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
    plt.show()

    layer1 = NeuronLayer(2,5) # lecture de la spirale (chaque point est décrit par x et y)
    activation1 = ActivationFunctions.ReLU()
    layer1.forward(X)
    print(layer1.output)
    activation1.forward(layer1.output)
    print(activation1.output)

    smLayer = NeuronLayer(2,3)
    acsm = ActivationFunctions.ReLU()
    smLayer2 = NeuronLayer(3,3)
    acsm2 = ActivationFunctions.Softmax()
    smLayer.forward(X)
    acsm.forward(smLayer.output)
    smLayer2.forward(acsm.output)
    acsm2.forward(smLayer2.output)
    print(acsm2.output)