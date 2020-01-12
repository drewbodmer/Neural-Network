import numpy as np
import random

LEARNING_SPEED = 0.15


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def cost(correct, output):
    y = create_y(correct)
    res = [0] * len(output)
    for x in range(len(output)):
        res[x] = (y[x] - output[x])**2
    res = np.asmatrix(res)
    return res


def ncost(correct, output):
    y = create_y(correct)
    res = np.subtract(y, output)
    res = sum(res) ** 2
    return res


def dcost(correct, output):
    y = create_y(correct)
    res = np.subtract(y, output) * LEARNING_SPEED
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


def prep_inputs(inputs):
    res = []
    for x in inputs:
        res.append(prep_input(x))
    return res


def generate(dim1, dim2):  # generates a dim1*dim2 array of random numbers between -1 and 1.
    res = []
    for x in range(dim1):
        row = [None] * dim2
        for y in range(dim2):
            row[y] = random.uniform(-1, 1)
        res.append(row)
    res = np.asmatrix(res)
    return res


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
