import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from array import array
# loading the file containing training data
data_file = input('enter file CSV filename : ')
data = pd.read_csv(data_file, header=None)
n1 = (len(data.columns))-1   # to determine the target value column from data file
Y = data.iloc[:, n1].values  # storing the target value
n = (len(data.columns)) - 1
X = data.iloc[:, 0:n].values  # storing all values of x in data file in one array

learning_rate = float(input('Enter the learning rate : '))
threshold = float(input('Enter the threshold rate : '))

length = len(X)  # length of data set
weights = np.zeros([n+1])   # initializing weights
# inserting x0 in the x array
x0 = np.ones(length)
X = np.insert(X, 0, x0, axis=1)   # inserting X0 values in X array as X0 is always 1

iters = 0  # to count iterations
sum_sq_error = 0
sqe_old = 1
sqe_new = 0
while (abs(sqe_new - sqe_old)) > threshold:
    learnt_func = []   # to store the learnt function after every weight update

    for i in range(length):
        a = 0
        for w, x in list(zip(weights, X[i])):
            a += w * x
        learnt_func.append(a)

    # calculating error between predicted and actual target
    for lf, y in zip(learnt_func, Y):
        error = (Y - learnt_func)

    # gradient descent to update weights based on errors obtained
    temp = []
    for w in range(len(weights)):
        gradient = sum(error * X[:, w])
        temp.append(weights[w] + learning_rate * gradient)

    # saving sum of squared errors
    sum_sq_error = 0

    for l, y in zip(learnt_func, Y):
        sum_sq_error += ((y-l) ** 2)

    sqe_new = sum_sq_error    # to check difference between new squared error and squared error in previous iterations
    print(iters, weights, sum_sq_error)

    weights = temp
    temp = []
    error = []
    if (abs(sqe_new - sqe_old)) < threshold:
        iters = length

    else:
        iters = iters + 1
        sqe_old = sqe_new
        sqe_new = 0






