from mlp_mestrado.neuron import Neuron


class Layer:
    def __init__(self, network, number_of_neurons):
        if len(network.layers) > 0:
            self.is_input_layer = False
            self.previous_layer = network.layers[-1]
        else:
            self.is_input_layer = True
            self.previous_layer = None

        self.neurons = []
        for iteration in range(number_of_neurons):
            neuron = Neuron(self.previous_layer)
            self.neurons.append(neuron)

    def think(self):
        for neuron in self.neurons:
            neuron.think(self.previous_layer)
