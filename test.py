import neuralnet as n
import load_data as ld
from PIL import Image


trainX, trainY, testX, testY = ld.load_dataset()
nn = n.NeuralNetwork(784, 16, 10)

nn.load_network()
train, test = ld.prep_pixels(trainX, testX)

correct = 0
for x in range(1000):
    predicted = nn.predict(test[x])
    actual = testY[x]
    # print("predicted: " + str(predicted) + " actual: " + str(actual))
    if predicted == actual:
        correct += 1
    else:
        image = Image.fromarray(testX[x])
        image.show()
        print("predicted: " + str(predicted) + " actual: " + str(actual))
        break
print("Accuracy: " + str(correct/10) + "%")


