class random_model:
    def __init__(self, n):
        self.n = n
    def __str__(self) -> str:
        return f'random_model with {self.n} nodes'
    def compute(self, *args, **kwargs):
        return np.random.rand(self.n)

class layer():
    def __init__(self, n):
        self.n = n
        self.is_mutateable = False

    def __str__(self) -> str:
        return f'layer with {self.n} nodes'

    def compile(self, *args, **kwkwargs):
        pass

    def compute(self, input : np.ndarray) -> np.ndarray:
        return sigmoid(input)
    
    def mutate(self):
        pass

class dense(layer):
    def compile(self, n_before : int, activation_func=plain):
        '''
        n_before the size of input or the size of output of previous layer
        function sets weights and biases for this layer''' 
        self.is_mutateable=True
        self.n_before = n_before
        self.weights = np.ones((self.n, n_before))
        self.biases = np.random.rand(self.n)/10
        self.activation_func = activation_func
    
    def compute(self, input) -> np.ndarray:
        computed = np.sum(input*self.weights, axis=1)# + self.biases
        return self.activation_func(computed)
    
    def to_json(self):
        return json.dumps(self)
    
    def mutate(self):
        '''mutation of weights and biases by a mean'''
        self.weights = add_random_n_places(self.weights, 5)
        self.biases = add_random_n_places(self.biases, 3)
class model:
    def __init__(self, layers=[]):
        self.layers=layers
    
    def __str__(self) -> str:
        return str([layer.__str__() for layer in self.layers])

    def __copy__(self):
        return model(self.layers)

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

def load_model(name='save') -> model:
    with open(str(name)+'.pkl', 'rb') as load_file:
        return pickle.load(load_file)
        
def generate_random_models(n:int)->list:
    '''generate random models'''
    models=[]
    for i in range(n):
        Model = model()
        Model.layers = [layer(9), dense(9)]
        Model.compile()
        Model.mutate()
        models.append(Model)
    return models
