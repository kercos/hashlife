from tqdm import tqdm
from gol.main_pure import init_gol_board_neighborhood_rule
from gol.main_pure import Automata
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

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

def test_cycle_fast(torch_device, iters=None):
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

    if iters is None:
        # single run
        if torch_device is not None:
            torch_operation(automata)
        else:
            numpy_operation(automata)
    else:
        # start_time = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("random_board"):
                # for i in range(iters):
                automata.set_random_board()
            with record_function("cycle_period"):
                automata.get_cycle_period()
        # end_time = time.time()
        # ellapsed_time = 1000 * (end_time - start_time) # milliseconds
        # print('ellapsed', ellapsed_time)

        print(prof.key_averages().table(sort_by="cpu_time_total")) # row_limit=10

        # for i in tqdm(range(tot_configurations)):
            # board = get_board_int(n=i, size=size)
            # automata.set_board(board)
            # automata.set_random_board()
            # automata.get_cycle_period()

if __name__ == "__main__":
    # torch_device = None
    torch_device = 'cuda'

    test_cycle_fast(
        torch_device = torch_device,
        iters = 1
    )

    '''
    single:
    numpy 135
    torch 107

    1000x:
    numpy 619
    torch 12338

    TODO:
    - rewrite python code so that things would go in a since cuda kernel (nvidia profile)
    - look into custom cuda kernel (cuda C)

    '''
