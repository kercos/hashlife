import time
from gol.utils import (
    init_gol_board_neighborhood_rule,
    numpy_to_life_106
)
from gol.hl.hashlife import construct

def generate_base(
        size,
        seed=123,
        file_life106=None
    ):
    '''Generate base (board, hl-node) from seed'''

    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        size = size,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = seed,
    )

    if file_life106 is not None:
        numpy_to_life_106(board, file_life106)

    # generate tuples (cells x,y coordinate which are 'on')
    pat_tuples = tuple(
        (x,y)
        for x in range(size)
        for y in range(size)
        if board[y,x]
    )

    # construct pattern
    init_t = time.process_time()
    node = construct(pat_tuples)
    t = time.process_time() - init_t
    print(f'Computation (hl-construct) took {t*1000.0:.1f} ms')

    return node, board, neighborhood, rule