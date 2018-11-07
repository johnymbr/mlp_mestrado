from random import seed
from mlp_mestrado import neural_network_m2 as nn

seed(1)

filename = 'result_matrix_ml.csv'
dataset = nn.load_csv(filename)
for i in range(len(dataset[0]) - 1):
    nn.str_column_to_float(dataset, i)

# convertendo coluna de classe para inteiros
lookup = nn.str_column_to_int(dataset, len(dataset[0]) - 1)
# normalizando dados
minmax = nn.dataset_minmax(dataset)
nn.normalize_dataset(dataset, minmax)
# avaliacao algoritmo
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 20
epsilon = 1e-06
scores, network = nn.evaluate_algorithm(dataset, nn.back_propagation, n_folds, n_hidden, l_rate, epsilon)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
print('Lookup: %s' % lookup)

filename_test = 'result_matrix_test.csv'
dataset_test = nn.load_csv(filename_test)
for i in range(len(dataset_test[0]) - 1):
    nn.str_column_to_float(dataset_test, i)

# convertendo coluna de classe para inteiros
lookup = nn.str_column_to_int(dataset_test, len(dataset_test[0]) - 1)
# normalizando dados
minmax = nn.dataset_minmax(dataset_test)
nn.normalize_dataset(dataset_test, minmax)

for row in dataset_test:
    prediction = nn.predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
