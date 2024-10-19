from math import floor, log
import numpy as np
from gol.main_pure import init_gol_board_neighborhood_rule
from gol.main_pure import Automata
from tqdm import tqdm

def get_board_int(n=15, size=4):
    total_size = size * size
    bin_str = np.binary_repr(n, width=total_size)
    a = [bool(int(bit)) for bit in bin_str]
    return np.reshape(a, (size, size))

def test_board(
        board_np,
        torus=True,
        use_fft = False,
        torch_device = None,
    ):

    size = board_np.shape[0]

    # init gol board and rule
    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        size = size,
        initial_state = board_np
    )

    # init automata
    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = torus,
        use_fft = use_fft,
        torch_device = torch_device,
    )

    automata.advance(iterations=100)

def test_cycles(
        size = 4,
        torus = True,
        use_fft = False,
        torch_device = None,
    ):

    shape = (size, size)
    total_cells = size * size
    max_states = (2 ** total_cells) - 1 # 15 when size is 2 (2x2=4)

    for state in tqdm(range(max_states)):
        board = get_board_int(n=state, size=size)

        test_board(
            board,
            torus = torus,
            use_fft = use_fft,
            torch_device = torch_device,
        )


if __name__ == "__main__":
    test_cycles(2)

    # a = get_board_int(n=15, size=2) # size=2 means 2x2 = 4 cells matrix with 16 configurations
    # print(a)