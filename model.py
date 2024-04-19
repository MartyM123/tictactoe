import numpy as np
import json
import pickle
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def plain(x):
    return x

def add_random_n_places(a, n):
    out = a.astype(float)

    # Generate unique flattened indices along the size of a
    idx = np.random.choice(a.size, n, replace=False)
    # Assign into those places ramdom numbers in [-1,1)
    out.flat[idx] += np.random.uniform(low=-1, high=1, size=n)
    return out

class layer():
    def __init__(self, n):
        self.n = n

    def __str__(self) -> str:
        return f'layer with {self.n} nodes'

    def compile(self, *args, **kwkwargs):
        pass

    def compute(self, input : np.ndarray) -> np.ndarray:
        return input
    
    def mutate(self):
        pass

class dense(layer):
    def compile(self, n_before : int, activation_func=plain):
        '''
        n_before the size of input or the size of output of previous layer
        function sets weights and biases for this layer''' 
        self.n_before = n_before
        self.weights = np.zeros((self.n, n_before))
        self.biases = np.zeros(self.n)
        self.activation_func = activation_func
    
    def compute(self, input) -> np.ndarray:
        computed = np.sum(input*self.weights, axis=1) + self.biases
        return self.activation_func(computed)
    
    def to_json(self):
        return json.dumps(self)
    
    def mutate(self):
        '''mutation of weights'''
        self.weights = add_random_n_places(self.weights, 5)


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

    def save(self, name='save'):
        with open(str(name)+'.pkl', 'wb') as save_file:
            pickle.dump(self, save_file)
    
    def mutate(self):
        for layer in self.layers:
            layer.mutate()

def load_model(name='save'):
    with open(str(name)+'.pkl', 'rb') as load_file:
        return pickle.load(load_file)

Model=model()
Model.layers=[layer(9), dense(9)]

Model.compile()

print(Model.compute(np.array([1,0,1,0,-1,-1,-1,0,1])))
Model.mutate()
print(Model.compute(np.array([1,0,1,0,-1,-1,-1,0,1])))