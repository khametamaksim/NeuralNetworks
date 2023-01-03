import numpy as np


class NeuralNetwork:

    def __init__(self, input_shape, shape, output_shape):
        self.W = {}
        self.B = {}
        self.Pre = {}
        self.A = {}
        self.shape = len(shape)
        self.sizes = [input_shape] + shape + [output_shape]
        for i in range(self.shape + 1):
            self.W[i + 1] = np.random.normal(0, 0.01, size=(self.sizes[i], self.sizes[i+1]))
            self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def total_error(self, predictions, y):
        sum = 0
        y = y.flat
        predictions = predictions.flat
        for index, prediction in enumerate(predictions):
            sum += (prediction - y[index])**2
        return sum / len(predictions)

    def forward_pass(self, X, y):
        predictions = []

        for x in X:
            self.Pre[0] = x.reshape(1, -1)
            for i in range(self.shape + 1):
                self.A[i + 1] = np.dot(self.Pre[i], np.asmatrix(self.W[i + 1])) + self.B[i + 1]
                if i + 1 != self.shape + 1:
                    self.Pre[i + 1] = self.sigmoid(self.A[i + 1])
            predictions.append(self.A[self.shape + 1])
        return self.total_error(np.array(predictions), y)

    def size(self):
        size = 0
        for i in range(self.shape + 1):
            size += self.sizes[i] * self.sizes[i + 1] + self.sizes[i + 1]
        return size

    def set_values(self, values):
        for i in range(self.shape + 1):
            self.W[i + 1] = values[:self.sizes[i] * self.sizes[i + 1]].reshape(self.W[i + 1].shape)
            values = np.delete(values, range(self.sizes[i] * self.sizes[i + 1]))

            self.B[i + 1] = values[:self.sizes[i + 1]].reshape(self.B[i + 1].shape)
            values = np.delete(values, range(self.sizes[i + 1]))
