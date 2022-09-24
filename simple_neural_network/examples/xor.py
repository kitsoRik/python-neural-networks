from simple_neural_network.lib.neural_network import NeuralNetwork
from random import shuffle


def activate(value):
    return [round(x[0]) for x in value]


nn = NeuralNetwork(2, 4, 1)

# training data
training_inputs = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
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
print(nn.feed_forward([0, 0]), "passed" if activate(nn.feed_forward([0, 0])) == [0] else "failed, expected " + str([0]) + ", got " + str(activate(nn.feed_forward([0, 0]))))
print(nn.feed_forward([0, 1]), "passed" if activate(nn.feed_forward([0, 1])) == [1] else "failed, expected " + str([1]) + ", got " + str(activate(nn.feed_forward([0, 1]))))
print(nn.feed_forward([1, 0]), "passed" if activate(nn.feed_forward([1, 0])) == [1] else "failed, expected " + str([1]) + ", got " + str(activate(nn.feed_forward([1, 0]))))
print(nn.feed_forward([1, 1]), "passed" if activate(nn.feed_forward([1, 1])) == [0] else "failed, expected " + str([0]) + ", got " + str(activate(nn.feed_forward([1, 1]))))
