import betterneuralnet as n
import load_data as ld
import utils
import random

trainX, trainY, testX, testY = ld.load_dataset()
train, test = ld.prep_pixels(trainX, testX)
nn = n.NeuralNetwork(784, 16, 10)

nn.initialize()
sample_size = len(trainY)

utils.printProgressBar(0, sample_size, prefix='Progress:', suffix='Complete', length=50)
for x in range(round(sample_size)):
    randomint = random.randint(0, len(train) - 20)
    batch = zip(trainY[randomint:(randomint+20)], utils.prep_inputs(train[randomint:(randomint + 20)]))
    utils.printProgressBar(0, sample_size, prefix='Progress:', suffix='Complete', length=50)
    nn.update_mini_batch(batch)
nn.save_network()


def testNN():
    nn.load_network()
    correct = 0
    traincorrect = 0
    for x in range(10000):
        predicted = nn.predict(test[x])
        trainpredict = nn.predict(train[x])
        actual = testY[x]
        trainact = trainY[x]
        if predicted == actual:
            correct += 1
        if trainpredict == trainact:
            traincorrect += 1

    print("Training accuracy: " + str(traincorrect / 100) + "%")
    print("Accuracy: " + str(correct / 100) + "%")


# testNN()
