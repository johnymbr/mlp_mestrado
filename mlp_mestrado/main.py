from mlp_mestrado import parameters
from mlp_mestrado.neural_network import NeuralNetwork
from numpy import random


class TrainingExample:
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output


def seed_random_generator():
    random.seed(1)


def calculate_average_error(p_cumulative_error, number_of_examples):
    if p_cumulative_error:
        return round(p_cumulative_error * 100 / number_of_examples, 2)
    else:
        return None


if __name__ == "__main__":
    seed_random_generator()

    network = NeuralNetwork([3, 4, 1])

    examples = [
        TrainingExample([0, 0, 1], 0),
        TrainingExample([0, 1, 1], 1),
        TrainingExample([1, 0, 1], 1),
        TrainingExample([0, 1, 0], 1),
        TrainingExample([1, 0, 0], 1),
        TrainingExample([1, 1, 1], 0),
        TrainingExample([0, 0, 0], 0),
    ]

    for i in range(1, parameters.training_iterations + 1):
        cumulative_error = 0
        for e, example in enumerate(examples):
            cumulative_error += network.train(example)
        average_error = calculate_average_error(cumulative_error, len(examples))

    new_situation = [1, 1, 0]
    print("Considering a new situation " + str(new_situation) + "?")
    print(network.think(new_situation))
