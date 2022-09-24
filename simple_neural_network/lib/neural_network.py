import numpy as np

class NeuralNetwork:
    def __init__(self, inputs_size: int, hidden_size: int, outputs_size: int):
        self.inputs_size = inputs_size
        self.hidden_size = inputs_size
        self.outputs_size = inputs_size

        self.ih_weights = [[] for _ in range(inputs_size)]
        self.ho_weights = [[] for _ in range(hidden_size)]

        self.bias_h = np.array([[np.random.rand()] for _ in range(hidden_size)])
        self.bias_o = np.array([[np.random.rand()] for _ in range(outputs_size)])

        for i in range(inputs_size):
            self.ih_weights[i] = [np.random.rand() for _ in range(hidden_size)]

        for i in range(hidden_size):
            self.ho_weights[i] = [np.random.rand() for _ in range(outputs_size)]

        self.ih_weights = np.array(self.ih_weights).T
        self.ho_weights = np.array(self.ho_weights).T

    def train(self, inputs_array, targets_array):
        inputs = np.array([inputs_array]).T
        targets = np.array([targets_array]).T

        _, hidden, outputs = self.__feed_forward__(inputs)

        errors = np.subtract(targets, outputs)

        self.feed_backward(inputs, hidden, outputs, errors, targets)

    def feed_backward(self, inputs, hidden, outputs, errors, targets):
        gradients = self.dsigmoid_map(outputs)
        gradients = np.multiply(gradients, errors)
        gradients = np.multiply(gradients, self.get_learning_rate())

        weights_ho_deltas = np.dot(gradients, hidden.T)

        self.ho_weights = np.add(self.ho_weights, weights_ho_deltas)
        self.bias_o = np.add(self.bias_o, gradients)

        hidden_errors = np.dot(self.ho_weights.T, errors)
        hidden_gradient = self.dsigmoid_map(hidden)
        hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
        hidden_gradient = hidden_gradient * self.get_learning_rate()

        weights_ih_deltas = np.dot(hidden_gradient, inputs.T)

        self.ih_weights = np.add(self.ih_weights, weights_ih_deltas)
        self.bias_h = np.add(self.bias_h, hidden_gradient)



    def feed_forward(self, inputs_array):
        inputs = np.array([inputs_array]).T

        _, _, outputs = self.__feed_forward__(inputs)

        return outputs

    def __feed_forward__(self, inputs):
        hidden = np.dot(self.ih_weights, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid_map(hidden)

        outputs = np.dot(self.ho_weights, hidden)
        outputs = np.add(outputs, self.bias_o)
        outputs = self.sigmoid_map(outputs)

        return [inputs, hidden, outputs]

    def sigmoid_map(self, inputs):
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                inputs[i][j] = 1 / (1 + np.exp(-inputs[i][j]))

        return inputs

    def dsigmoid_map(self, inputs):
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                inputs[i][j] = inputs[i][j] * (1 - inputs[i][j])

        return inputs

    def get_learning_rate(self):
        return 0.01
