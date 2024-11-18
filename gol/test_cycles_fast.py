from tqdm import tqdm
from gol.main_pure import init_gol_board_neighborhood_rule
from gol.main_pure import Automata


def test_cycle_fast(torch_device):
    size = 4
    total_cells = size * size
    tot_configurations = 2 ** total_cells

    board, neighborhood, rule = init_gol_board_neighborhood_rule(size)

    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = True,
        use_fft = False,
        torch_device = torch_device,
    )

    for i in tqdm(range(tot_configurations)):
        # board = get_board_int(n=i, size=size)
        # automata.set_board(board)
        automata.set_random_board()
        automata.get_cycle_period()


if __name__ == "__main__":
    test_cycle_fast(
        torch_device = 'mps'
    )
