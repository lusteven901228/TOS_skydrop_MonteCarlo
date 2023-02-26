import numpy as np
from skimage.measure import label
import time

default_board = np.array([[1,1,1,2,2,2],
                          [3,3,3,4,4,4],
                          [5,5,5,6,6,6],
                          [1,1,1,2,2,2],
                          [3,3,3,4,4,4]], dtype=np.uint8)

result = []

n = 1_000_000 # 10,000 loops takes about 2.8s on 3070Ti
iterator = iter(range(n))

def step(board):
    
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

    return label(board2, background=0, return_num=True, connectivity=1)[1]

def simulate():
    board = np.copy(default_board)
    combos = 0
    while i:=step(board):
        combos += i
    return combos

if __name__ == "__main__":
    result = []
    t = time.time()
    result = [simulate() for _ in range(n)]
    bin_range = (max(result)) - min(result)
    print(np.histogram(result, bins=bin_range))
    print(time.time()-t)
    # t = time.time()
    # print(sum((simulate() for _ in range(10000)))/10000)
    # print(time.time()-t)
