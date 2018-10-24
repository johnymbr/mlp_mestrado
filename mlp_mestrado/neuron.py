import mlp_mestrado.neuron_synapse as ns


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


