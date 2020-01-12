import betterneuralnet as n
import load_data as ld
import utils

trainX, trainY, testX, testY = ld.load_dataset()
nn = n.NeuralNetwork(784, 16, 10)

nn.load_network()
train, test = ld.prep_pixels(trainX, testX)


def testNN():
    correct = 0
    traincorrect = 0
    for x in range(10000):
        utils.printProgressBar(x, 10000, prefix='Progress:', suffix='Complete', length=50)
        predicted = nn.predict(test[x])
        trainpredict = nn.predict(train[x])
        actual = testY[x]
        trainact = trainY[x]
        # print("predicted: " + str(predicted) + " actual: " + str(actual))
        # image = Image.fromarray(testX[x])
        # image.show()
        # print("predicted: " + str(predicted) + " actual: " + str(actual))
        if predicted == actual:
            correct += 1
        # else:
        #     image = Image.fromarray(testX[x])
        #     image.show()
        #     print("predicted: " + str(predicted) + " actual: " + str(actual))
        if trainpredict == trainact:
            traincorrect += 1

    print("Training data accuracy: " + str(traincorrect/100) + "%")
    print("Test data accuracy: " + str(correct/100) + "%")


testNN()
# nn.print_weights()
# nn.display_network()

