import utils
import numpy as np


class NeuralNetwork:
    first_layer_wts = []
    first_layer_bias = []
    second_layer_wts = []
    second_layer_bias = []
    result_wts = []
    result_bias = []

    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

    def initialize(self):
        self.first_layer_wts = np.random.rand(self.input_size, self.hidden_layer_size) * 2 - 1
        self.first_layer_bias = np.random.rand(self.hidden_layer_size) * 2 - 1

        self.second_layer_wts = np.random.rand(self.hidden_layer_size, self.hidden_layer_size) * 2 - 1
        self.second_layer_bias = np.random.rand(self.hidden_layer_size) * 2 - 1

        self.result_wts = np.random.rand(self.hidden_layer_size, self.output_size) * 2 - 1
        self.result_bias = np.random.rand(self.output_size) * 2 - 1

    def feed_forward(self, inputs, norm_func):
        """
        Calculates the activation of each layer
        :param inputs: A 28*28 matrix of values representing pixel activations.
        :return: The activations of each of the three layers.
        """
        pix = [0] * len(inputs) ** 2
        count = 0
        for x in inputs:
            for y in x:
                pix[count] = y
                count += 1

        first_layer_activation = norm_func(np.add(np.matmul(np.asmatrix(pix), self.first_layer_wts),
                                                  self.first_layer_bias.transpose()))  # 1*784 X 784*16

        second_layer_activation = norm_func(np.add(np.matmul(first_layer_activation, self.second_layer_wts),
                                                   self.second_layer_bias.transpose()))  # 1*16 X 16*16

        result_activation = norm_func(np.add(np.matmul(second_layer_activation, self.result_wts),
                                             self.result_bias.transpose()))

        return np.asarray(first_layer_activation)[0], \
            np.asarray(second_layer_activation)[0], np.asarray(result_activation)[0]

    def backpropagate(self, inputs, correct):
        first_layer_activation, second_layer_activation, result_activation = self.feed_forward(inputs, utils.sigmoidarr)
        first_layer_z, second_layer_z, result_z = self.feed_forward(inputs, utils.dsigmoidarr)
        cost = utils.dcost(correct, result_activation)
        error = np.multiply(cost, result_z)
        print(error)
