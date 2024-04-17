import numpy as np

class layer():
    def __init__(self, n):
        self.n = n

    def __str__(self) -> str:
        return f'layer with {self.n} nodes'

    def compile(self, *args, **kwkwargs):
        pass

    def compute(self, input : np.ndarray) -> np.ndarray:
        return input

class dense(layer):
    def compile(self, n_before : int):
        '''
        n_before the size of input or the size of output of previous layer
        function sets weights and biases for this layer''' 
        self.n_before = n_before
        self.weights = (np.ones(self.n*n_before)/4).reshape((self.n, n_before))
        self.biases = np.ones(self.n)/10
    
    def compute(self, input) -> np.ndarray:
        computed = np.sum(input*self.weights, axis=1)
        return computed

class model:
    def __init__(self, layers=[]):
        self.layers=layers
    
    def __str__(self) -> str:
        return str([layer.__str__() for layer in self.layers])

    def compute(self, input : np.ndarray) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.compute(output)
        return output
    
    def compile(self):
        for i in range(len(self.layers)-1):
            self.layers[i+1].compile(n_before=self.layers[i].n)

    
Model=model()
Model.layers=[layer(3), dense(4), dense(3)]

Model.compile()

Dense = dense(4)
Dense.compile(3)

print(Model.compute(np.array([1,2,3])))