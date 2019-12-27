import neuralnet as n
import load_data as ld
import utils

trainX, trainY, testX, testY = ld.load_dataset()
nn = n.NeuralNetwork(784, 16, 10)

nn.initialize()
train, test = ld.prep_pixels(trainX, testX)

for x in range(60000):
    nn.backpropagate(train[x], trainY[x])
    nn.save_network()
