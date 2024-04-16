import numpy as np

class layer():
    def __init__(self, n):
        self.n = n

class input_layer(layer):
    def compute(input : np.ndarray) -> np.ndarray:
        return input

class dense(layer):
    def compile(self, n_before : int):
        '''
        n_before the size of input or the size of output of previous layer
        function sets weights and biases for this layer''' 
        self.n_before = n_before
        self.weights = (np.ones(self.n*n_before)/4).reshape((n_before, self.n))
        self.biases = np.ones(self.n)/10
    
    def compute(self, input) -> np.ndarray:
        computed = np.sum(input*self.weights, axis=1)
        return np.where(computed>self.biases, computed, 0) #bias filter
    

Dense=dense(9)
Dense.compile(9)
print(Dense.compute(np.array([1,0,0,-1,-1,0,1,1,0])))