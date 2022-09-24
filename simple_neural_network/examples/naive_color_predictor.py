# native color predictor
# that example show how to predict the color of a pixel based on its RGB values
# the neural network will be trained with a set of RGB values and their corresponding color
# the neural network will then be tested with a set of RGB values and the neural network will try to predict the color of the pixel

from simple_neural_network.lib.neural_network import NeuralNetwork
from random import shuffle

def activate(value):
    return [round(x[0]) for x in value]


nn = NeuralNetwork(3, 6, 3)

# data consists of
# 0 - amount of red color in the pixel
# 1 - amount of green color in the pixel
# 2 - amount of blue color in the pixel

# training data
training_inputs = [
    [[1, 0, 0], [1, 0, 0]],
    [[0, 1, 0], [0, 1, 0]],
    [[0, 0, 1], [0, 0, 1]],
    [[1, 1, 0], [1, 1, 0]],
    [[1, 0, 1], [1, 0, 1]],
    [[0, 1, 1], [0, 1, 1]],
    [[1, 1, 1], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 0]],
    [[0.5, 0.5, 0.5], [1, 1, 1]],
    [[0.4, 0.7, 0.3], [0, 1, 0]],
    [[0.4, 0.3, 0.3], [0, 0, 0]],
    [[0.4, 0.9, 0.8], [0, 1, 1]],
    [[0.7, 0.4, 0.8], [1, 0, 1]],
    [[0.55, 0.65, 0.45], [1, 1, 0]],
]

print("# Initializing ")

epochs_amount = 10000

for epoch in range(epochs_amount):
    # randomize array
    shuffle(training_inputs)

    for inputs, expected in training_inputs:
        nn.train(inputs, expected)

    # print progress
    if epoch % (epochs_amount / 100 * 5) == 0:
        print("# Progress ", epoch, "/", epochs_amount, "|", "{0:0.0f}".format(epoch / epochs_amount * 100), "/ 100%")

# test output for real data
# red
print(nn.feed_forward([1, 0, 0]), "passed" if activate(nn.feed_forward([1, 0, 0])) == [1, 0, 0] else "failed, expected " + str([1, 0, 0]) + ", got " + str(activate(nn.feed_forward([1, 0, 0]))))
# green
print(nn.feed_forward([0, 1, 0]), "passed" if activate(nn.feed_forward([0, 1, 0])) == [0, 1, 0] else "failed, expected " + str([0, 1, 0]) + ", got " + str(activate(nn.feed_forward([0, 1, 0]))))
# blue
print(nn.feed_forward([0, 0, 1]), "passed" if activate(nn.feed_forward([0, 0, 1])) == [0, 0, 1] else "failed, expected " + str([0, 0, 1]) + ", got " + str(activate(nn.feed_forward([0, 0, 1]))))
# yellow
print(nn.feed_forward([1, 1, 0]), "passed" if activate(nn.feed_forward([1, 1, 0])) == [1, 1, 0] else "failed, expected " + str([1, 1, 0]) + ", got " + str(activate(nn.feed_forward([1, 1, 0]))))
# purple
print(nn.feed_forward([1, 0, 1]), "passed" if activate(nn.feed_forward([1, 0, 1])) == [1, 0, 1] else "failed, expected " + str([1, 0, 1]) + ", got " + str(activate(nn.feed_forward([1, 0, 1]))))
# cyan
print(nn.feed_forward([0, 1, 1]), "passed" if activate(nn.feed_forward([0, 1, 1])) == [0, 1, 1] else "failed, expected " + str([0, 1, 1]) + ", got " + str(activate(nn.feed_forward([0, 1, 1]))))
# white
print(nn.feed_forward([1, 1, 1]), "passed" if activate(nn.feed_forward([1, 1, 1])) == [1, 1, 1] else "failed, expected " + str([1, 1, 1]) + ", got " + str(activate(nn.feed_forward([1, 1, 1]))))
# black
print(nn.feed_forward([0, 0, 0]), "passed" if activate(nn.feed_forward([0, 0, 0])) == [0, 0, 0] else "failed, expected " + str([0, 0, 0]) + ", got " + str(activate(nn.feed_forward([0, 0, 0]))))
# gray
print(nn.feed_forward([0.5, 0.5, 0.5]), "passed" if activate(nn.feed_forward([0.5, 0.5, 0.5])) == [1, 1, 1] else "failed, expected " + str([1, 1, 1]) + ", got " + str(activate(nn.feed_forward([0.5, 0.5, 0.5]))))
# greenish
print(nn.feed_forward([0.2, 0.59, 0.1]), "passed" if activate(nn.feed_forward([0.2, 0.59, 0.1])) == [0, 1, 0] else "failed, expected " + str([0, 1, 0]) + ", got " + str(activate(nn.feed_forward([0.2, 0.59, 0.1]))))
# blackish
print(nn.feed_forward([0.2, 0.2, 0.2]), "passed" if activate(nn.feed_forward([0.2, 0.2, 0.2])) == [0, 0, 0] else "failed, expected " + str([0, 0, 0]) + ", got " + str(activate(nn.feed_forward([0.2, 0.2, 0.2]))))
# cyanish
print(nn.feed_forward([0.2, 0.9, 0.8]), "passed" if activate(nn.feed_forward([0.2, 0.9, 0.8])) == [0, 1, 1] else "failed, expected " + str([0, 1, 1]) + ", got " + str(activate(nn.feed_forward([0.2, 0.9, 0.8]))))
