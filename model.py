import numpy as np
import json
import pickle
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def plain(x):
    return x

def add_random_n_places(a : np.ndarray, n : int) -> np.ndarray:
    '''add random values to random n elements'''
    out = a.astype(float)

    # Generate unique flattened indices along the size of a
    idx = np.random.choice(a.size, n, replace=False)
    # Assign into those places ramdom numbers in [-1,1)
    out.flat[idx] += np.random.uniform(low=-1, high=1, size=n)
    return out

def get_max_exc(a : np.ndarray, exc=[]) -> int:
    '''return index of max element except indecies in exc:list'''
    m = np.zeros(a.size, dtype=bool)
    m[exc] = True
    b = np.ma.array(a, mask=m)
    return int(np.argmax(b))

class layer():
    def __init__(self, n):
        self.n = n
        self.is_mutateable = False

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
        self.is_mutateable=True
        self.n_before = n_before
        self.weights = np.zeros((self.n, n_before))
        self.biases = np.random.rand(self.n)/10
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

    def reproduce(self, parents, include_self=False):
        '''sets weights and biases of all mutateable layers to average of parents
        if include_self is set to True one of parents is model its self'''
        if include_self: parents.append(self)
        n=len(parents)
        for i_layer in range(len(self.layers)):
            if self.layers[i_layer].is_mutateable:
                w=np.zeros(self.layers[i_layer].weights.shape)
                b=np.zeros(self.layers[i_layer].biases.shape)
                for parent in parents:
                    w=w+parent.layers[i_layer].weights
                    b=b+parent.layers[i_layer].biases
                self.layers[i_layer].weights=w/n
                self.layers[i_layer].biases=b/n

def load_model(name='save') -> model:
    with open(str(name)+'.pkl', 'rb') as load_file:
        return pickle.load(load_file)

def is_full(board:np.ndarray)->bool:
    if 0 not in board:
        return True
    return False

def is_winner(board:np.ndarray)->bool:
    n=3
    board=np.reshape(board,(n,n))
    res=(np.sum(board,axis=0), np.sum(board,axis=1), np.trace(board), np.trace(board[::-1]))
    res=np.hstack(res)
    if n in res or -n in res:
        return True
    return False

def is_legal(board:np.ndarray, i:int):
    '''Check if move to i index is legal on board'''
    if board[i]==0:
        return True
    return False

def execute(board:np.ndarray, prediction:np.ndarray)->np.ndarray:
    '''play the most popular (based on predicton) legal move and return the board after the play'''
    exc=[]
    while True:
        request=get_max_exc(prediction, exc)
        if is_legal(board, request):
            board[request]+=1
            return board
        else:
            exc.append(request)

def fight(model1:model, model2:model, show=False)->list:
    '''returns who won the match
            first won: [1,0]
            second won: [0,1]
            tie: [0,0]'''
    board = np.zeros(9)
    flip=1
    while True:
        #first model is 1 second is -1 empty tile is 0
        for m in [model1, model2]:
            prediction=m.compute(board)
            board = execute(board, prediction)
            if is_winner(board):
                if flip==1:
                    if show: print(board.reshape((3,3))*flip)
                    return [1,0]
                if show: print(board.reshape((3,3))*flip)
                return [0,1]
            elif is_full(board):
                #tie
                if show: print(board.reshape((3,3))*flip)
                return [0,0]
            #flip the board
            board*=-1
            flip*=-1

def score(a:list)->list:
    '''return the overall score of each model'''
    table=np.zeros((len(models), len(models)))
    for i,model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:]):
            j+=i+1
            res=fight(model1, model2)
            table[i][j]+=res[0]
            table[j][i]+=res[1]

            res=fight(model2, model1)
            table[i][j]+=res[0]
            table[j][i]+=res[1]

    return np.sum(table, axis=1)

def choose_parents(models:list, score:list)->list:
    '''choose two parents with gratest score and return them in list'''
    zipped_sorted = sorted(zip(models,score), key=lambda x:x[1], reverse=True)[:2]
    return [a for a,b in zipped_sorted]
        

models=[]

for i in range(10):
    Model = model()
    Model.layers = [layer(9), dense(9)]
    Model.compile()
    Model.mutate()
    models.append(Model)

m=model()
m.layers=[layer(9), dense(9)]
m.compile()

m.reproduce(models)

