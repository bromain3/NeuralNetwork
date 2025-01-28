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
x_train_formated = np.array(x_train_formated)
y_train = np.array(y_train)
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

dense1 = nl.NeuronLayer(784, 256)
activation1 = af.ReLU()
dense2 = nl.NeuronLayer(256, 256)
activation2 = af.ReLU()
dense3 = nl.NeuronLayer(256, 10)
activation3 = lf.CombinedSoftmaxCrossEntropy()
optimizer = opt.StochasticGradientDescent(learningRate=3,decay=1e-3,momentum=0.8)
accuracy = lf.Accuracy()

L = []
for i in range(int(60000/32)):
    dense1.forward(x_train_formated[32*i:32*(i+1)])
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = activation3.forward(dense3.output, y_train[32*i:32*(i+1)])
    acc = accuracy.calculate(activation3.output, y_train[32*i:32*(i+1)])
    #print(acsm2.output, loss)

    activation3.backward(activation3.output, y_train[32*i:32*(i+1)])
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

    if i%100 == 0:
        print('loss:', loss)
        print('acc:', acc)
        print('epoch:', i)
        print('learning rate:', optimizer.currentLearningRate)
        L.append(loss)


plt.plot(L)
plt.show()