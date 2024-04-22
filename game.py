from model import *

Model=load_model('model2')


'''returns who won the match
        first won: [1,0]
        second won: [0,1]
        tie: [0,0]'''
board = np.zeros((3,3))

print('You start and you have -1')
while True:
    c=int(input("column: "))
    r=int(input("row: "))
    board[r][c]-=1
    if is_winner(board):
        print('you won')
        break
    elif is_full(board):
        print('tie')
        break

    board=np.hstack(board)
    predictions=Model.compute(board)
    board=execute(board, predictions)
    board=board.reshape((3,3))
    
    print(board)
    if is_winner(board):
        print('AI won')
        break
    elif is_full(board):
        print('tie')
        break
