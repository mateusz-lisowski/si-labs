from collections import defaultdict
import numpy as np
from node_solution import Node


class DecisionTree:
    def __init__(self, params):
        self.root_node = Node()
        self.params = defaultdict(lambda: None, params)

    def train(self, array, y):
        self.root_node.train(array, y, self.params)

    def evaluate(self, array, y):
        predicted = self.predict(array)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, array):
        prediction = []
        for x in array:
            prediction.append(self.root_node.predict(x))
        return prediction

