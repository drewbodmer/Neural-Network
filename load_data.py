from keras.datasets import mnist
import tensorflow

def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm