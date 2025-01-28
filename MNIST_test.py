import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath


    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = 'MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1
    plt.show()


mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images
#

x_train_formated = []
for i in x_train:
    x_train_formated.append(np.array(i).flatten())
x_train_formated = np.array(x_train_formated).astype('float64')
x_train_formated -= (np.max(x_train_formated) + np.min(x_train_formated)) / 2
x_train_formated /= (np.max(x_train_formated) - np.min(x_train_formated)) / 2
print(np.max(x_train_formated), np.min(x_train_formated))
y_train = np.array(y_train)

keys = np.array(range(x_train_formated.shape[0]))
np.random.shuffle(keys)
x_train_formated = x_train_formated[keys]
y_train = y_train[keys]

images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)


import NeuronLayer as nl
import ActivationFunctions as af
import LossFunctions as lf
import Optimizer as opt
import NeuralNetwork as nn


neural = nn.NeuralNetwork(784, 10, [30,30])
neural.setActivationFunction(0, af.ReLU())
neural.setActivationFunction(1, af.ReLU())
neural.setActivationFunction(2, lf.CombinedSoftmaxCrossEntropy())
neural.setAcuracyFunction(lf.SoftmaxClassificationAccuracy())
#neural.setOptimizerFunction(opt.AdaptativeMomentum(learningRate=0.005, decay=1e-3)) # 97.4% accuracy for 10 epochs with big batch for each period (two 128 neuron hidden layers)
neural.setOptimizerFunction(opt.AdaptativeMomentum(learningRate=0.005, decay=1e-3))  # 96.08% accuracy for 30-50 epochs with big batch for each period (two 30 neuron hidden layers)


L, A, Lr = neural.train(x_train_formated, y_train, 50,bigBatchSize='all')




x_test_formated = []
for i in x_test:
    x_test_formated.append(np.array(i).flatten())
x_test_formated = np.array(x_test_formated).astype('float64')
x_test_formated -= (np.max(x_test_formated) + np.min(x_test_formated)) / 2
x_test_formated /= (np.max(x_test_formated) - np.min(x_test_formated)) / 2
print(np.max(x_test_formated), np.min(x_test_formated))
y_test = np.array(y_test)

result, loss, acc = neural.testNetwork(x_test_formated, y_test)

print(result, loss, acc)

titles_2_show.append('test image [' + str(0) + '] = ' + str(y_test[0]))
plt.imshow(x_test[0], cmap=plt.cm.gray)
plt.show()
print(list(result[0]), np.max(result[0]), list(result[0]).index(max(result[0])))

"""
dense1 = nl.NeuronLayer(784, 256)
activation1 = af.ReLU()
dense2 = nl.NeuronLayer(256, 256)
activation2 = af.ReLU()
dense3 = nl.NeuronLayer(256, 10)
activation3 = lf.CombinedSoftmaxCrossEntropy()
optimizer = opt.AdaptativeMomentum(learningRate=0.02, decay=1e-5)
accuracy = lf.SoftmaxClassificationAccuracy()

BATCH_SIZE = 128
steps = x_train_formated.shape[0] // BATCH_SIZE
if steps * BATCH_SIZE < x_train_formated.shape[0]:
    steps += 1
epoch = 10


L = []
for i in range(epoch):
    for step in range(steps):
        batchX = x_train_formated[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        batchy = y_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        dense1.forward(batchX)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        loss = activation3.forward(dense3.output, batchy)
        acc = accuracy.calculate(activation3.output, batchy)
        #print(acsm2.output, loss)

        activation3.backward(activation3.output, batchy)
        dense3.backward(activation3.inputGradient)
        activation2.backward(dense3.inputGradient)
        dense2.backward(activation2.inputGradient)
        activation1.backward(dense2.inputGradient)
        dense1.backward(activation1.inputGradient)
        #print(smLayer.inputGradient)

        optimizer.preUpdateParams()
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
        optimizer.updateParams(dense3)
        optimizer.postUpdateParams()

        if step%100 == 0:
            print('loss:', loss)
            print('acc:', acc)
            print('epoch:', i)
            print('step:', step)
            print('learning rate:', optimizer.currentLearningRate)
            L.append(loss)
    '''
    dense1.forward(x_train_formated)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = activation3.forward(dense3.output, y_train)
    acc = accuracy.calculate(activation3.output, y_train)
    # print(acsm2.output, loss)

    activation3.backward(activation3.output, y_train)
    dense3.backward(activation3.inputGradient)
    activation2.backward(dense3.inputGradient)
    dense2.backward(activation2.inputGradient)
    activation1.backward(dense2.inputGradient)
    dense1.backward(activation1.inputGradient)
    # print(smLayer.inputGradient)

    optimizer.preUpdateParams()
    optimizer.updateParams(dense1)
    optimizer.updateParams(dense2)
    optimizer.updateParams(dense3)
    optimizer.postUpdateParams()
    '''
"""

plt.plot(L)
plt.show()
plt.plot(A)
plt.show()
plt.plot(Lr)
plt.show()
