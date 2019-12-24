import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoidarr(x):
    x = np.asarray(x)[0]
    res = [None] * len(x)
    for i in range(len(x)):
        res[i] = 1 / (1 + math.exp(-x[i]))
    return np.asmatrix(res)


def dsigmoid(x):
    return 1 / (1 + math.exp(-x)) * (1 - (1 / (1 + math.exp(-x))))


def dsigmoidarr(x):
    x = np.asarray(x)[0]
    res = [None] * len(x)
    for i in range(len(x)):
        res[i] = 1 / (1 + math.exp(-x[i])) * (1 - (1 / (1 + math.exp(-x[i]))))
    return np.asmatrix(res)



def cost(correct, output):
    y = create_y(correct)
    res = [0] * len(output)
    for x in range(len(output)):
        res[x] = (y[x] - output[x])**2
    return res


def dcost(correct, output):
    y = create_y(correct)
    res = [0] * len(output)
    for x in range(len(output)):
        res[x] = (y[x] - output[x])
    return res


def create_y(num):
    res = [0] * 10
    res[num] = 1
    return res