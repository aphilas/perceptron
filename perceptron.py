from typing import Tuple
 
def dot_product(l1, l2):
  return sum(i*j for i,j in zip(l1, l2))

class Perceptron:
  def __init__(self, threshold=0.5, learning_rate=0.1, bias=0):
    self.threshold = threshold
    self.learning_rate = learning_rate
    self.bias = 0

  def __activation(self, x):
    return 1 if x >= self.threshold else 0

  def __weighted_sum(self, values, weights):
    return dot_product(values, weights)

  def train(self, training_set: Tuple[Tuple[int, ...], int], weights, epochs=50, max_error=0):
    self.weights = weights
    epoch_count = 0

    while True:
      for values, output in training_set:
        ws = self.__weighted_sum(values, weights)
        result = self.__activation(ws) > self.threshold
        error = output - result

        if error != 0:
          for index, value in enumerate(values):
            self.weights[index] += self.learning_rate * error * value

      if epoch_count >= epochs:
        break

      print(f'Epoch {epoch_count}')
      epoch_count += 1

  def predict(self, values):
    return self.__weighted_sum(values, self.weights)