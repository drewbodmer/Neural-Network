import neuralnet as n
import load_data as ld
import utils

trainX, trainY, testX, testY = ld.load_dataset()
nn = n.NeuralNetwork(784, 16, 10)

nn.initialize()
train, test = ld.prep_pixels(trainX, testX)
nn.backpropagate(train[0], trainY[0])