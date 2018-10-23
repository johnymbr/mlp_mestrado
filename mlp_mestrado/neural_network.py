##
## https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a
## https://medium.com/deep-learning-101/how-to-generate-a-video-of-a-neural-network-learning-in-python-62f5c520e85c
##
import numpy as np


class NeuralNetwork:
    def __init__(self, number_of_layers):
        self.number_of_layers = number_of_layers
