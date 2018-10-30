##
## https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a
## https://medium.com/deep-learning-101/how-to-generate-a-video-of-a-neural-network-learning-in-python-62f5c520e85c
##
from mlp_mestrado.neuron_layer import Layer


class NeuralNetwork:
    def __init__(self, requested_layers):
        self.layers = []
        for number_of_neurons in requested_layers:
            self.layers.append(Layer(self, number_of_neurons))

    def train(self, example):
        error = example.output - self.think(example.inputs)
        self.reset_errors()
        self.layers[-1].neurons[0].error = error
        for l in range(len(self.layers) - 1, 0, -1):
            for neuron in self.layers[l].neurons:
                self.layers[l - 1] = neuron.train(self.layers[l - 1])
        return fabs(error)

    def do_not_think(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.output = 0
                for synapse in neuron.synapses:
                    synapse.signal = 0

    def think(self, inputs):
        for layer in self.layers:
            if layer.is_input_layer:
                for index, value in enumerate(inputs):
                    self.layers[0].neurons[index].output = value
            else:
                layer.think()
        return self.layers[-1].neurons[0].output

    def reset_errors(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.error = 0
