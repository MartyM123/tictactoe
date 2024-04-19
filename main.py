import numpy as np
import pygad


board=np.zeros(9).reshape((3,3))

def isFull(board):
    if 0 not in board:
        return True
    return False

def isWinner(board):
    board=board.reshape((3,3))
    n=3
    res=(np.sum(board,axis=0), np.sum(board,axis=1), np.trace(board), np.trace(board[::-1]))
    res=np.hstack(res)
    if n in res or -n in res:
        return True
    return False



def simple_comp(board):
    pass

def check_and_update(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i, j] == 0:
                board[i, j] = -1
                return board
    return board

# Call the function

new_board = check_and_update(board)
print(new_board)
board = new_board


print(isWinner(board))