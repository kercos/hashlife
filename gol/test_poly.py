from matplotlib import pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm
from gol.pure.automata import Automata
from gol.main_pure import init_gol_board_neighborhood_rule
import torch

def life_update(x,y):
    '''
    Credits to Raymond Baranski (Alex)
    `x` is the cell-state (dead=0, alive=1), and
    `y` is the number of alive neighbors.
    We only care about the sign information of the polynomial (to keep things simple, otherwise the polynomial would be much more complex). Numpy sign returns -1 and 1, so I shift it (add 1) and normalize (divide by 2) to get it back to 0 and 1
    f(x,y) = -1*(1-x)*(y-2.5)*(y-3.5)-x*(y-1.5)*(y-3.5)
    f(a,b,c,d,x,f,g,h,i) =
       -(a+b+c+d+f+g+h+i)**2 - x*(a+b+c+d+f+g+h+i) + 3.5*x + 6*(a+b+c+d+f+g+h+i) - 8.75
       =
       -a**2 - f**2 - 2*a*b - 2*a*c - 2*a*d - 2*a*f - 2*a*g - 2*a*h - 2*i*a - b**2 - 2*b*c - 2*b*d - 2*b*f - 2*b*g - 2*b*h - 2*i*b - c**2 - 2*c*d - 2*c*f - 2*c*g - 2*c*h - 2*i*c - d**2 - i**2 - 2*d*f - 2*d*g - 2*d*h - 2*i*d - 2*f*g - 2*f*h - 2*i*f - g**2 - 2*g*h - 2*i*g - h**2 - 2*i*h - a*x - b*x - c*x - d*x - f*x - g*x - h*x - i*x + 3.5*x + 6*a + 6*b + 6*c + 6*d + 6*f + 6*g + 6*h + 6*i - 8.75
    '''
    return (
        np.sign(-y**2 - x*y + 3.5*x + 6*y - 8.75)
        +1
    ) /2

def life_update_torch(x,y):
    return (
        # np.sign(-1*(1-x)*(y-2.5)*(y-3.5)-x*(y-1.5)*(y-3.5))
        torch.sign(-y**2 - x*y + 3.5*x + 6*y - 8.75)
        +1
    ) /2

def poly_advance(board, neighborhood):
    counts_int = scipy.signal.convolve2d(
        board,
        neighborhood,
        mode = 'same',
        boundary = 'fill' # 'circular' if torus, 'fill' if strict
    )
    new_board = life_update(board, counts_int)
    return new_board, counts_int

def test(size, seed):

    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        size = size,
        initial_state = 'random',
        density = 0.5,
        seed = seed
    )

    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = False,
        use_fft = False,
        torch_device = None,
    )

    # automata.show_current_frame('test 0', force_show=False)
    board_0 = automata.get_board_numpy(change_to_int=True)
    automata.advance(1)
    # automata.show_current_frame('test 1', force_show=False)
    board_1 = automata.get_board_numpy(change_to_int=True)
    poly_board_1, counts_int = poly_advance(board_0, neighborhood)
    test = np.all(board_1==poly_board_1)
    if not test:
        # plt.show()
        print('Test failed!')
        print('board 0')
        print(board_0)
        print('board 1')
        print(board_1)
        print('poly counts int')
        print(counts_int)
        print('poly board 1')
        print(poly_board_1)
    return test



if __name__ == "__main__":
    size = 10
    for i in tqdm(range(1000)):
        if not test(size=3, seed=1):
            break