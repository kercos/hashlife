from matplotlib import pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm
from gol.pure.automata import Automata
from gol.main_pure import init_gol_board_neighborhood_rule

def poly_solve(x,y):
    '''
    Credits to Raymond Baranski (Alex)
    `x` is the cell-state (dead=0, alive=1), and
    `y` is the number of alive neighbors.
    We only care about the sign information of the polynomial (to keep things simple, otherwise the polynomial would be much more complex). Numpy sign returns -1 and 1, so I shift it (add 1) and normalize (divide by 2) to get it back to 0 and 1
    '''
    return int(
        (
            np.sign(-1*(1-x)*(y-2.5)*(y-3.5)-x*(y-1.5)*(y-3.5))
            +1
        )
        /2
    )

    return

def poly_advance(board, neighborhood):
    counts_int = scipy.signal.convolve2d(
            board,
            neighborhood,
            mode = 'same',
            boundary = 'fill' # 'circular' if torus, 'fill' if strict
        )
    size = board.shape[0]
    counts_int = np.rint(counts_int).astype(int)
    new_board = np.zeros_like(counts_int, dtype=int)
    for i in range(size):
        for j in range(size):
            x = board[i,j]
            y = counts_int[i,j]
            new_board[i,j] = poly_solve(x,y)
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
    size = 4
    for i in tqdm(range(1000)):
        if not test(size=3, seed=1):
            break