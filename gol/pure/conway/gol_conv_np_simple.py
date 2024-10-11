import numpy as np
from scipy.ndimage import convolve

'''
cells: ?
'''
def life(cells):
    result = convolve(cells, k, mode="wrap")
    return (result>4) & (result<8)

if __name__ == "__main__":
    k = np.array([1,1,1,1,2,1,1,1,1]).reshape(3,3)
    life(5)