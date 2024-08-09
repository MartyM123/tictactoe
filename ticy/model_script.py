# %% [markdown]
# # Hunt for perfect model
# 
# This is file which have my thoughts and content about training neural networks for tictactoe.

# %%
import numpy as np
import pickle

# %% [markdown]
# ### General model knowledge
# model will get 2d numpy array where 1 means bots marks, 0 empty title and -1 the opponents title. For now consider only boards 3x3.
# As the output of the model should return row, column where to play.

# %% [markdown]
# ### Brain of model
# Brain will consist different parts each part will handle some part of game decision.
# - First Part will consist where to play to win
# - Second part will consist where to play to not lose
# - Third part will consist where to play to have biggest profit on it (strategy)

# %% [markdown]
# # First Part
# --------------------------------

# %%
def check_horizontal(board):
    for i, row in enumerate(board):
        if np.sum(row)==2:
            return np.where(row==0)[0].item(), i
        
def check_vertical(board):
    for i in range(board.shape[1]):
        if np.sum(board[:, i]) == 2:
            return i, np.where(board[:, i] == 0)[0].item()

def check_diagonal(board):
    diagonal1 = np.diagonal(board)
    if np.sum(diagonal1) == 2:
        return np.where(diagonal1 == 0)[0].item(), np.where(diagonal1 == 0)[0].item()
    diagonal2 = np.diagonal(np.fliplr(board))
    if np.sum(diagonal2) == 2:
        return np.where(np.flip(diagonal2) == 0)[0].item(), np.where(diagonal2 == 0)[0].item()

# %%
def first_part(board):
    if check_diagonal(board):
        return check_diagonal(board)
    if check_horizontal(board):
        return check_horizontal(board)
    if check_vertical(board):
        return check_vertical(board)

# %% [markdown]
# # Second part - check blocks
# ----------------------------------------------------------------

# %%
board = np.array([[0,0,0],
                  [0,1,0],
                  [1,0,0]])

# %%
def check_block_horizontal(board):
    for i, row in enumerate(board):
        if np.sum(row)==-2:
            return np.where(row==0)[0].item(), i
        
def check_block_vertical(board):
    for i in range(board.shape[1]):
        if np.sum(board[:, i]) == -2:
            return i, np.where(board[:, i] == 0)[0].item()

def check_block_diagonal(board):
    diagonal1 = np.diagonal(board)
    if np.sum(diagonal1) == -2:
        return np.where(diagonal1 == 0)[0].item(), np.where(diagonal1 == 0)[0].item()
    diagonal2 = np.diagonal(np.fliplr(board))
    if np.sum(diagonal2) == -2:
        return np.where(np.flip(diagonal2) == 0)[0].item(), np.where(diagonal2 == 0)[0].item()

# %%
def second_part(board):
    if check_block_diagonal(board):
        return check_block_diagonal(board)
    if check_block_horizontal(board):
        return check_block_horizontal(board)
    if check_block_vertical(board):
        return check_block_vertical(board)

# %% [markdown]
# # Model compose
# ----------------------------------------------------------------

# %%
def load_model(name='model'):
    with open(str(name)+'.pkl', 'rb') as load_file:
        return pickle.load(load_file)

# %%
def print_output(func):
    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        print(result)
        return result
    return wrap

# %%
class Model:
    def __init__(self, name='model'):
        self.name=name

    def save(self, name):
        with open(str(name)+'.pkl', 'wb') as save_file:
            pickle.dump(self, save_file)
    
    @print_output
    def play(self, board):
        if first_part(board):
            return first_part(board)
        if second_part(board):
            return second_part(board)
        else:
            return np.where(board==0)[0][0], np.where(board==0)[1][0]
            


# %%



