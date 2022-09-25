# handwritten digits
# that example tries to predict what digit is written on the image
# using mnist dataset
# https://en.wikipedia.org/wiki/MNIST_database
# with 60 000 training images and 10 000 test images
# each image is 28x28 pixels
# each pixel is represented by a number from 0 to 255 (0 - black, 255 - white)
# each image is represented by a list of 784 numbers (28 * 28)
# each number is a pixel
# with error ~6%

from keras.datasets import mnist

from simple_neural_network.lib.neural_network import NeuralNetwork
import numpy as np

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

def activate(value):
    max_value = max(value)

    return [1 if x == max_value else 0 for x in value]


nn = NeuralNetwork(784, 196, 10)

nn.set_learning_rate(0.1)

print("# Initializing ")

epochs_amount = int(len(train_Y))

for epoch in range(epochs_amount):
    # randomize array
    label = train_Y[epoch % len(train_Y)]
    image_matrix = train_X[epoch % len(train_Y)]
    flatten_image_matrix = np.concatenate(np.array(image_matrix))

    inputs = [x / 255 for x in flatten_image_matrix]

    outputs = [0] * 10
    outputs[label] = 1

    got_outputs = nn.train(inputs, outputs)

    # print progress
    if epoch % ((epochs_amount) / 100) == 0:
        print("# Progress ", epoch, "/", epochs_amount, "|", "{0:0.0f}".format(epoch / epochs_amount * 100), "/ 100%")

correct_amount = 0
# test output with test data
for idx in range(len(test_Y)):
    label = test_Y[idx]
    image_matrix = test_X[idx]
    flatten_image_matrix = np.concatenate(np.array(image_matrix))
    inputs = [x / 255 for x in flatten_image_matrix]

    expected_outputs = [0] * 10
    expected_outputs[label] = 1

    outputs = nn.feed_forward(inputs).T[0]

    if expected_outputs == activate(outputs):
        print("For " + str(label) + " Correct")
        correct_amount += 1
    else:
        print("For " + str(label) + " Incorrect")

print("Correct", correct_amount)
print("Incorrect", len(test_Y) - correct_amount)
print("Error approx ", (len(test_Y) - correct_amount) / len(test_Y) * 100, "%")
