import numpy as np
import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoidarr(x):
    res = [None] * len(x)
    for i in range(len(x)):
        res[i] = 1 / (1 + math.exp(-x[i]))
    res = np.asmatrix(res).transpose()
    return res


def dsigmoid(x):
    return 1 / (1 + math.exp(-x)) * (1 - (1 / (1 + math.exp(-x))))


def dsigmoidarr(x):
    res = [None] * len(x)
    for i in range(len(x)):
        res[i] = 1 / (1 + math.exp(-x[i])) * (1 - (1 / (1 + math.exp(-x[i]))))
    res = np.asmatrix(res).transpose()
    return res


def cost(correct, output):
    y = create_y(correct)
    res = [0] * len(output)
    for x in range(len(output)):
        res[x] = (y[x] - output[x])**2
    res = np.asmatrix(res).transpose()
    return res


def dcost(correct, output):
    y = create_y(correct)
    res = np.subtract(y, output)
    return res


def create_y(num):
    res = [0] * 10
    res[num] = 1
    res = np.asmatrix(res).transpose()
    return res


def prep_input(inputs):
    pix = [0] * len(inputs) ** 2
    count = 0
    for x in inputs:
        for y in x:
            pix[count] = y
            count += 1
    pix = np.asmatrix(pix).transpose()
    return pix


def generate(dim1, dim2):  # generates a dim1*dim2 array of random numbers between -1 and 1.
    res = []
    for x in range(dim1):
        row = [None] * dim2
        for y in range(dim2):
            row[y] = random.uniform(-1, 1)
        res.append(row)
    res = np.asmatrix(res)
    return res
