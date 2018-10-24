import numpy as np


class NeuronSynapse:
    def __init__(self, input_neuron_index):
        self.input_neuron_index = input_neuron_index
        self.weight = random_weight()
        self.signal = 0


def random_weight():
    return 2 * np.random.random() - 1
