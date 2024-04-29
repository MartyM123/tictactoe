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

def is_legal(board:np.ndarray, i:int) -> bool:
    '''Check if move to i index is legal on board'''
    if board[i]==0:
        return True
    return False

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
            if show:
                print('palyer: '+str(flip))
                print(board.reshape((3,3))*flip)
                print()

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