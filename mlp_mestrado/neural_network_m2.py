from random import random, randrange
import numpy as np
from csv import reader
import matplotlib.pyplot as plt

# Carregando um arquivo CSV
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Converte coluna de string em float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Converte coluna de string em int
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Encontrar os valores minimo e maximo para cada coluna do dataset
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Normalizando os dados entre 0 e 1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Repartindo o arquivo para aplicar k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Percentual de acuracia
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Avalia algoritmo de aprendizagem usando cross validation
def evaluate_algorithm(dataset, algorithm, n_folds, n_layers, n_hidden, l_rate, epsilon):
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, n_layers, n_hidden, n_outputs)

    print_layers(network)

    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(network, train_set, test_set, n_outputs, l_rate, epsilon)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    print_layers(network)
    return scores, network


# Iniciando a rede
def initialize_network(n_inputs, n_layers, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    for i in range(n_layers - 1):
        hidden_layer = [{'weights': [random() for i in range(len(hidden_layer) + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calcula a ativacao do neuronio para um dado
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Funcao de transferencia
def transfer(activation):
    # return 1.0 / (1.0 + np.exp(-activation))
    return np.tanh(activation)


# Funcao de transferencia derivada
def transfer_derivative(output):
    # return output * (1.0 - output)
    return 1 - (output ** 2)


# Propagando o dado para a saida da rede
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Retropropagacao do erro e  salvando nos neuronios
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# atualizar os pesos da rede
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Treinando a rede por um numero fixo de epocas
def train_network(network, train, l_rate, epsilon, n_outputs):
    epoch = 0
    eqm_prev = None
    errorsqm = []
    while True:
        epoch += 1
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

        eqm_current = sum_error / len(train)

        errorsqm.append(eqm_current)

        if eqm_prev is None:
            eqm_prev = eqm_current
        else:
            if epoch % 500 == 0:
                print('>>>', abs(eqm_current - eqm_prev), ' ', epsilon, ' | epoch: ', epoch)

            if abs(eqm_current - eqm_prev) <= epsilon:
                print('>>>', abs(eqm_current - eqm_prev), ' ', epsilon)
                print('converged in epoch >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ', epoch)

                # Plota como o erro se comportou em relacao as epocas
                plt.plot(errorsqm)
                plt.xlabel('Epocas')
                plt.ylabel('EQM')
                plt.show()

                break
            eqm_prev = eqm_current

        # print_layers(network)
        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Predicao com a rede treinada
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Printa as camadas da rede
def print_layers(network):
    for layer in network:
        print(layer)


#
def back_propagation(network, train, test, n_outputs, l_rate, epsilon):
    train_network(network, train, l_rate, epsilon, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions
