import numpy as np
import pandas as pd
from skimage.measure import label
import time

default_board = np.array([[1,1,1,2,2,2],
                          [3,3,3,4,4,4],
                          [5,5,5,6,6,6],
                          [1,1,1,2,2,2],
                          [3,3,3,4,4,4]], dtype=np.uint8)

result = []

n = 1_000_000 # 10,000 loops takes about 3.1s on 3070Ti
iterator = iter(range(n))

def dissolve(board):
    
    # decide what to dissolve
    dissolve_arr = np.zeros((5,6), dtype=bool)
    for i in range(1,4):
        for j in range(6):
            if board[i][j] == board[i-1][j] == board[i+1][j]:
                dissolve_arr[i][j] = True
                dissolve_arr[i-1][j] = True
                dissolve_arr[i+1][j] = True
    
    for i in range(5):
        for j in range(1,5):
            if board[i][j] == board[i][j-1] == board[i][j+1]:
                dissolve_arr[i][j] = True
                dissolve_arr[i][j-1] = True
                dissolve_arr[i][j+1] = True
    
    board2 = np.copy(board)
    board2[dissolve_arr==0]= 0

    board[dissolve_arr] = 0
    for i in range(board.shape[1]):
        col = board[:, i]
        non_zero_idx = np.nonzero(col)[0]
        n = len(non_zero_idx)
        col[0:n] = col[non_zero_idx]
        col[n:] = 255
    board[board == 255] = np.random.randint(1, 7, size=np.count_nonzero(board == 255))

    if label(board2, background=0, return_num=True, connectivity=1)[1]: 
        return dissolve(board)
    
    return np.sort(
        np.array([np.count_nonzero(board==i) for i in range(1,7)], dtype=np.uint8)
        )[::-1]
    
    
def simulate():
    board = np.copy(default_board)
    return dissolve(board)

if __name__ == "__main__":
    result = []
    t = time.time()
    result = [simulate() for _ in range(n)]
    print(sum((np.all(i>2) for i in result))/n) # type: ignore    
    # bin_range = (max(result)) - min(result)
    # print(np.histogram(result, bins=bin_range))
    df = pd.DataFrame(result, columns=[*'123456'])
    df.sort_values(by=[*'123456'], ascending=False, inplace=True)
    df.to_csv('results.csv', index=False)

    print(time.time()-t)
    
    # t = time.time()
    # print(sum((simulate() for _ in range(10000)))/10000)
    # print(time.time()-t)