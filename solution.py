import sys
import numpy as np
from network import NeuralNetwork
from algorithm import GeneticAlgorithm


def load_data(file):
    with open(file, 'r') as f:
        X, y = [], []
        for line in f:
            if line.startswith("x"):
                continue
            else:
                values = [float(e) for e in line.replace(",", " ").split()]
                X.append(values[:-1])
                y.append(values[-1])

    X, y = np.array(X), np.array(y)

    return X, y


if __name__ == "__main__":
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--train":
            train_file = sys.argv[i + 1]
        if sys.argv[i] == "--test":
            test_file = sys.argv[i + 1]
        if sys.argv[i] == "--nn":
            if sys.argv[i + 1] == "5s":
                nn = [5]
            elif sys.argv[i + 1] == "20s":
                nn = [20]
            else:
                nn = [5, 5]
        if sys.argv[i] == "--popsize":
            pop_size = int(sys.argv[i + 1])
        if sys.argv[i] == "--elitism":
            elitism = int(sys.argv[i + 1])
        if sys.argv[i] == "--p":
            probability = float(sys.argv[i + 1])
        if sys.argv[i] == "--K":
            scale = float(sys.argv[i + 1])
        if sys.argv[i] == "--iter":
            iterations = int(sys.argv[i + 1])

    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    NN = NeuralNetwork(X_train.shape[1], nn, 1)


    def error_function(v):
        NN.set_values(v)
        return NN.forward_pass(X_train, y_train)


    GA = GeneticAlgorithm(NN.size(),
                          error_function,
                          elitism=elitism,
                          populationSize=pop_size,
                          mutationProbability=probability,
                          mutationScale=scale,
                          numIterations=iterations)

    done = False
    while not done:
        done, i, best = GA.step()
        if i % 2000 == 0:
            print("[Train error @" + str(i) + "]: " + str(round(error_function(best), 6)))
        NN.set_values(best)

    test_error = NN.forward_pass(X_test, y_test)
    print("[Test error]: " + str(round(test_error, 6)))
