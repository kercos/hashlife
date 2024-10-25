from matplotlib import pyplot as plt
import numpy as np
import scipy
from gol.pure.automata import Automata
from gol.main_pure import init_gol_board_neighborhood_rule

def poly_solve(x,y):
    '''
    Credits to Raymond Baranski (Alex)
    `x` is the cell-state (dead=0, alive=1), and
    `y` is the number of alive neighbors.
    We only care about the sign information of the polynomial (to keep things simple, otherwise the polynomial would be much more complex). Numpy sign returns -1 and 1, so I shift it (add 1) and normalize (divide by 2) to get it back to 0 and 1
    '''
    return -1*(1-x)*(y-2.5)*(y-3.5)-x*(y-1.5)*(y-3.5)

def poly_advance(board, neighborhood):
    counts_int = scipy.signal.convolve2d(
            board,
            neighborhood,
            mode = 'same',
            boundary = 'fill' # 'circular' if torus, 'fill' if strict
        )
    size = board.shape[0]
    counts_int = np.rint(counts_int)
    print('poly counts int')
    print(counts_int)
    new_board = np.zeros_like(counts_int)
    for i in range(size):
        for j in range(size):
            x = board[i,j]
            y = counts_int[i,j]
            new_board[i,j] = poly_solve(x,y)
    print('poly board 1')
    print(new_board)

def test():

    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        size = 4,
        initial_state = 'random',
        density = 0.5,
        seed = 123
    )

    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = False,
        use_fft = False,
        torch_device = None,
    )

    automata.show_current_frame('test 0', force_show=False)
    board_0 = automata.get_board_numpy(change_to_int=True)
    automata.advance(1)
    automata.show_current_frame('test 1', force_show=False)
    board_1 = automata.get_board_numpy(change_to_int=True)
    print('board 0')
    print(board_0)
    print('board 1')
    print(board_1)
    # plt.show()
    poly_advance(board_0, neighborhood)

if __name__ == "__main__":
    test()