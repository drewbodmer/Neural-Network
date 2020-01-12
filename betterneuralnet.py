import utils
import numpy as np
import display


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
        self.learning_speed = 0.5

    def initialize(self):
        self.first_layer_wts = utils.generate(self.hidden_layer_size, self.input_size)
        self.first_layer_bias = utils.generate(self.hidden_layer_size, 1)

        self.second_layer_wts = utils.generate(self.hidden_layer_size, self.hidden_layer_size)
        self.second_layer_bias = utils.generate(self.hidden_layer_size, 1)

        self.result_wts = utils.generate(self.output_size, self.hidden_layer_size)
        self.result_bias = utils.generate(self.output_size, 1)

    def feed_forward(self, inputs):
        """
        Calculates the activation of each layer
        :param inputs: A matrix of values representing pixel activations.
        :return: The activations of each of the three layers.
        """
        flz, fla = self.calculate_layer(inputs, self.first_layer_wts, self.first_layer_bias)
        slz, sla = self.calculate_layer(fla, self.second_layer_wts, self.second_layer_bias)
        rz, ra = self.calculate_layer(sla, self.result_wts, self.result_bias)

        return flz, slz, rz

    def activations(self, flz, slz, rz):
        return utils.sigmoid(flz), utils.sigmoid(slz), utils.sigmoid(rz)

    def backprop(self, correct, inputs):
        flz, slz, rz = self.feed_forward(inputs)
        result_error = self.result_error(correct, inputs)
        second_layer_error = self.error(self.result_wts, result_error, slz)
        first_layer_error = self.error(self.second_layer_wts, second_layer_error, flz)

        result_wts = np.matmul(result_error, utils.sigmoid(slz).transpose())
        second_layer_wts = np.matmul(second_layer_error, utils.sigmoid(flz).transpose())
        first_layer_wts = np.matmul(first_layer_error, utils.sigmoid(inputs).transpose())

        return result_wts, result_error, second_layer_wts, second_layer_error, first_layer_wts, first_layer_error

    def update_mini_batch(self, batch):
        first_layer_wt = np.zeros(self.first_layer_wts.shape)
        first_layer_bias = np.zeros(self.first_layer_bias.shape)
        second_layer_wt = np.zeros(self.second_layer_wts.shape)
        second_layer_bias = np.zeros(self.second_layer_bias.shape)
        result_wt = np.zeros(self.result_wts.shape)
        result_bias = np.zeros(self.result_bias.shape)

        for correct, inputs in batch:
            nrw, nrb, nslw, nslb, nflw, nflb = self.backprop(correct, inputs)

            first_layer_wt = np.add(first_layer_wt, nflw)
            first_layer_bias = np.add(first_layer_bias, nflb)
            second_layer_wt = np.add(second_layer_wt, nslw)
            second_layer_bias = np.add(second_layer_bias, nslb)
            result_wt = np.add(result_wt, nrw)
            result_bias = np.add(result_bias, nrb)

        self.first_layer_wts = np.add(self.first_layer_wts, first_layer_wt)
        self.first_layer_bias = np.add(self.first_layer_bias, first_layer_bias)
        self.second_layer_wts = np.add(self.second_layer_wts, second_layer_wt)
        self.second_layer_bias = np.add(self.second_layer_bias, second_layer_bias)
        self.result_wts = np.add(self.result_wts, result_wt)
        self.result_bias = np.add(self.result_bias, result_bias)

    def calculate_layer(self, inputs, wt, bias):
        z = np.add(np.matmul(wt, inputs), bias)
        activation = utils.sigmoid(z)
        return z, activation

    def result_error(self, correct, inputs):
        flz, slz, rz = self.feed_forward(inputs)
        cost = utils.ncost(correct, utils.sigmoid(rz))
        # self.learning_speed = cost
        return np.multiply(utils.dcost(correct, utils.sigmoid(rz)), utils.dsigmoid(rz))

    def error(self, wt, error, z):
        return np.multiply(np.matmul(wt.transpose(), error), utils.dsigmoid(z))

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

    def print_weights(self):
        print("result weights")
        print(self.result_wts)
        print(self.result_bias)
        print(self.first_layer_wts)
        print(self.first_layer_bias)

    def display_network(self):
        display.draw_graph(self.second_layer_wts)
