from tqdm import tqdm
from gol.main_pure import init_gol_board_neighborhood_rule
from gol.main_pure import Automata
import torch
import time

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def torch_operation(automata):
    start.record()
    automata.set_random_board()
    automata.get_cycle_period()
    end.record()
    torch.cuda.synchronize()
    ellapsed = start.elapsed_time(end) # milliseconds
    print('torch', ellapsed)

def numpy_operation(automata):
    start = time.time()
    automata.set_random_board()
    automata.get_cycle_period()
    end = time.time()
    ellapsed = 1000 * (end - start) # milliseconds
    print('numpy', ellapsed)

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
        use_poly_update = True,
        torch_device = torch_device,
    )

    # for i in tqdm(range(tot_configurations)):
        # board = get_board_int(n=i, size=size)
        # automata.set_board(board)
        # automata.set_random_board()
        # automata.get_cycle_period()

    '''
    start_time = time.time()

    for i in range(100):
        automata.set_random_board()
        automata.get_cycle_period()

    end_time = time.time()
    ellapsed_time = 1000 * (end_time - start_time) # milliseconds
    print('ellapsed', ellapsed_time)
    '''

    if torch_device is not None:
        torch_operation(automata)
    else:
        numpy_operation(automata)

if __name__ == "__main__":
    test_cycle_fast(
        torch_device = None
    )
    test_cycle_fast(
        torch_device = 'cuda'
    )
