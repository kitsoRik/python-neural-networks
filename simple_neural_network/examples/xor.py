from simple_neural_network.lib.neural_network import NeuralNetwork
from random import shuffle

nn = NeuralNetwork(2, 2, 1)

# training data
training_inputs = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
]

print("# Initializing ")

epochs_amount = 50000

for epoch in range(epochs_amount):
    shuffle(training_inputs)

    for inputs, expected in training_inputs:
        nn.train(inputs, expected)

    # print progress
    if epoch % 1000 == 0:
        print("# Progress ", epoch, "/", epochs_amount, "|", "{0:0.0f}".format(epoch / epochs_amount * 100), "/ 100%")

# test output for real data
print(nn.feed_forward([0, 0]))  # should be zero
print(nn.feed_forward([0, 1]))  # should be one
print(nn.feed_forward([1, 0]))  # should be one
print(nn.feed_forward([1, 1]))  # should be zero
