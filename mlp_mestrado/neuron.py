import mlp_mestrado.neuron_synapse as ns
import numpy as np


class Neuron:
    def __init__(self, previous_layer):
        self.output = 0
        self.synapses = []
        self.error = 0
        index = 0
        if previous_layer:
            for input_neuron in previous_layer.neurons:
                synapse = ns.NeuronSynapse(index)
                self.synapses.append(synapse)
                index += 1

    def train(self, previous_layer):
        for synapse in self.synapses:
            previous_layer.neurons[synapse.input_neuron_index].error += \
                self.error * sigmoid_derivative(self.output) * synapse.weight
            synapse.weight += synapse.signal * self.error * sigmoid_derivative(self.output)

    def think(self, previous_layer):
        activity = 0
        for synapse in self.synapses:
            synapse.signal = previous_layer.neurons[synapse.input_neuron_index].output
            activity += synapse.weight * synapse.signal
        self.output = sigmoid(activity)


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def sigmoid_derivative(x):
    return x * (1 - x)
