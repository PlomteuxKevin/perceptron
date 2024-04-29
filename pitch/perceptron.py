import numpy as np

class p():
    def __init__(self, xi:np.array):
        self.xs = np.array(xi)
        self.dataSize = self.xs.size
        self.b = 0
        self.ws = np.zeros(self.dataSize)
        self.y = 0

    def __fagr(self) -> float: # Agragate Function
        y = 0
        for x, w in zip(self.xs, self.ws):
            y += x * w
        return y + self.b

    def __fact(self, x:float) -> int: # Activation Function : Heaviside
        return 1 if x > 0 else 0

    def predict(self, xi:np.array):
        self.xs = xi
        self.y = self.__fact(self.__fagr())
