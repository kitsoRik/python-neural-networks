# whether to lend
# that example tries to predict whether to lend money to a person
# based on his current work, amount of his loans in the past, amount of overdue loans in the past
# and whether he has a loan right now

from simple_neural_network.lib.neural_network import NeuralNetwork
from random import shuffle

def activate(value):
    return [round(x[0]) for x in value]


nn = NeuralNetwork(4, 6, 1)

# data consists of
# 0 - person works
# 1 - person had at least one loan in the past
# 2 - person had at least one overdue loan
# 3 - person has a loan now

# training data
training_inputs = [
    [[1, 0, 0, 0], [1]],  # works, no loans, no overdue, no loan now
    [[1, 1, 0, 0], [1]],  # works, has loans, no overdue, no loan now
    [[1, 1, 1, 0], [0]],  # works, has loans, has overdue, no loan now
    [[1, 1, 1, 1], [0]],  # works, has loans, has overdue, has loan now
    [[0, 1, 1, 0], [0]],  # does not work, has loans, has overdue, no loan now
    [[0, 1, 1, 1], [0]],  # does not work, has loans, has overdue, has loan now
    [[1, 1, 0, 0], [1]],  # works, has loans, no overdue, no loan now
    [[1, 1, 0, 1], [0]],  # works, has loans, no overdue, has loan now
    [[0, 1, 0, 0], [0]],  # does not work, has loans, no overdue, no loan now
]

print("# Initializing ")

epochs_amount = 40000

for epoch in range(epochs_amount):
    # randomize array
    shuffle(training_inputs)

    for inputs, expected in training_inputs:
        nn.train(inputs, expected)

    # print progress
    if epoch % (epochs_amount / 100 * 5) == 0:
        print("# Progress ", epoch, "/", epochs_amount, "|", "{0:0.0f}".format(epoch / epochs_amount * 100), "/ 100%")

# test output for real data
# works, no loans, no overdue, no loan now
print(nn.feed_forward([1, 0, 0, 0]), "passed" if activate(nn.feed_forward([1, 0, 0, 0])) == [1] else "failed, expected " + str([1]) + ", got " + str(activate(nn.feed_forward([1, 0, 0, 0]))))
# does not work, no loans, no overdue, no loan now
print(nn.feed_forward([0, 0, 0, 0]), "passed" if activate(nn.feed_forward([0, 0, 0, 0])) == [0] else "failed, expected " + str([0]) + ", got " + str(activate(nn.feed_forward([0, 0, 0, 0]))))
# works, has loans, no overdue, no loan now
print(nn.feed_forward([1, 1, 0, 0]), "passed" if activate(nn.feed_forward([1, 1, 0, 0])) == [1] else "failed, expected " + str([1]) + ", got " + str(activate(nn.feed_forward([1, 1, 0, 0]))))
# works, has loans, has overdue, no loan now
print(nn.feed_forward([1, 1, 1, 0]), "passed" if activate(nn.feed_forward([1, 1, 1, 0])) == [0] else "failed, expected " + str([0]) + ", got " + str(activate(nn.feed_forward([1, 1, 1, 0]))))
# works, no loans, no overdue, has loan now
print(nn.feed_forward([1, 0, 0, 1]), "passed" if activate(nn.feed_forward([1, 0, 0, 1])) == [0] else "failed, expected " + str([0]) + ", got " + str(activate(nn.feed_forward([1, 0, 0, 1]))))
