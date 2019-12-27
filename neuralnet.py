import utils
import numpy as np


class NeuralNetwork:
    first_layer_wts = np.asmatrix([])
    first_layer_bias = np.asmatrix([])
    second_layer_wts = np.asmatrix([])
    second_layer_bias = np.asmatrix([])
    result_wts = np.asmatrix([])
    result_bias = np.asmatrix([])

    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

    def initialize(self):
        self.first_layer_wts = utils.generate(self.input_size, self.hidden_layer_size)
        self.first_layer_bias = utils.generate(self.hidden_layer_size, 1)

        self.second_layer_wts = utils.generate(self.hidden_layer_size, self.hidden_layer_size)
        self.second_layer_bias = utils.generate(self.hidden_layer_size, 1)

        self.result_wts = utils.generate(self.hidden_layer_size, self.output_size)
        self.result_bias = utils.generate(self.output_size, 1)

    def feed_forward(self, inputs):
        """
        Calculates the activation of each layer
        :param norm_func: The function used to normalize the output.
        :param inputs: A 28*28 matrix of values representing pixel activations.
        :return: The activations of each of the three layers.
        """

        first_layer_activation = utils.sigmoidarr(np.add(np.matmul(self.first_layer_wts.transpose(), inputs),
                                                         self.first_layer_bias))  # 16*784 X 784*1 = 16*1

        second_layer_activation = utils.sigmoidarr(
            np.add(np.matmul(self.second_layer_wts.transpose(), first_layer_activation),
                   self.second_layer_bias))  # 16*16 X 16*1 = 16*1

        result_activation = utils.sigmoidarr(np.add(np.matmul(self.result_wts.transpose(), second_layer_activation),
                                                    self.result_bias))  # 16*1 X 16*10 = 16*10

        return first_layer_activation, second_layer_activation, result_activation

    def backpropagate(self, inputs, correct):
        pix = utils.prep_input(inputs)
        first_layer_activation, second_layer_activation, result_activation = self.feed_forward(pix)
        first_layer_z, second_layer_z, result_z = self.calculate_z_values(pix)

        dcost = utils.dcost(correct, result_activation)
        # print(dcost)
        output_error = np.multiply(dcost, result_z)
        # print(output_error)
        second_layer_error = np.multiply(np.matmul(self.result_wts, output_error), second_layer_z)
        first_layer_error = np.multiply(np.matmul(self.second_layer_wts, second_layer_error), first_layer_z)

        self.result_wts = np.add(self.result_wts, np.matmul(second_layer_activation, output_error.transpose()))
        self.second_layer_wts = np.add(self.second_layer_wts, np.matmul(first_layer_activation, second_layer_error.transpose()))
        self.first_layer_wts = np.add(self.first_layer_wts, np.matmul(pix, first_layer_error.transpose()))

        self.result_bias = np.add(self.result_bias, output_error)
        self.second_layer_bias = np.add(self.second_layer_bias, second_layer_error)
        self.first_layer_bias = np.add(self.first_layer_bias, first_layer_error)

    # def backprop(self):

    def calculate_z_values(self, inputs):
        first_layer_activation, second_layer_activation, result_activation = self.feed_forward(inputs)
        first_layer_z = utils.dsigmoidarr(np.add(np.matmul(self.first_layer_wts.transpose(), inputs),
                                                 self.first_layer_bias))  # 16*784 X 784*1 = 16*1

        second_layer_z = utils.dsigmoidarr(np.add(np.matmul(self.second_layer_wts.transpose(), first_layer_activation),
                                                  self.second_layer_bias))  # 16*16 X 16*1 = 16*1

        result_z = utils.dsigmoidarr(np.add(np.matmul(self.result_wts.transpose(), second_layer_activation),
                                            self.result_bias))  # 16*1 X 16*10 = 16*10

        return first_layer_z, second_layer_z, result_z

    def predict(self, inputs):
        pix = utils.prep_input(inputs)
        fa, sa, result = self.feed_forward(pix)
        return result.argmax()

    def save_network(self):
        np.save("flw", self.first_layer_wts)
        np.save("slw", self.second_layer_wts)
        np.save("rlw", self.result_wts)
        np.save("flb", self.first_layer_bias)
        np.save("slb", self.second_layer_bias)
        np.save("rlb", self.result_bias)

    def load_network(self):
        self.first_layer_wts = np.load("flw.npy")
        self.second_layer_wts = np.load("slw.npy")
        self.result_wts = np.load("rlw.npy")
        self.first_layer_bias = np.load("flb.npy")
        self.second_layer_bias = np.load("slb.npy")
        self.result_bias = np.load("rlb.npy")
