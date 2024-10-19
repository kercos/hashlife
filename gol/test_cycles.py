from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from gol.main_pure import init_gol_board_neighborhood_rule
from gol.main_pure import Automata
from gol.utils import numpy_to_stars
from gol.process_lexicon import get_lex_patterns

NAME_PATTERN = get_lex_patterns()
PATTERN_NAME = {p:n for n,p in NAME_PATTERN.items()}

def get_board_int(n=15, size=4):
    total_size = size * size
    bin_str = np.binary_repr(n, width=total_size)
    a = [bool(int(bit)) for bit in bin_str]
    return np.reshape(a, (size, size))

def get_board_seed(seed=123, size=4, density=0.5):
    shape = (size, size)
    rng = np.random.default_rng(seed)
    board = rng.uniform(0, 1, shape)
    board = board < density
    return board

def get_min_on_cells(board_cycle):
    counts_on_cells = [
        np.count_nonzero(board_np)
        for board_np in board_cycle
    ]

    return np.min(counts_on_cells)

def print_patterns(board_cycle, all=False):
    if all:
        for c,b in enumerate(board_cycle, start=1):
            pattern_str = '\n'.join(numpy_to_stars(b, crop=True))
            print(f'-{c}-')
            print(pattern_str)
            print()
    else:
        first_board = board_cycle[0]
        pattern_str = '\n'.join(numpy_to_stars(first_board, crop=True))
        print(pattern_str)

def identify_pattern(board_cycle):
    for b in board_cycle:
        pattern_str = '\n'.join(numpy_to_stars(b, crop=True))
        if pattern_str in PATTERN_NAME:
            name = PATTERN_NAME[pattern_str]
            print('FOUND PATTERN:', name)
            return name
    return None

def get_board_cycle_period(
        size = 2,
        init_state = 7, # int (binary representation of matrix) or seed (random matrix)
        use_random_seed = False,
        iterations = 100,
        torus=True,
        use_fft = False,
        torch_device = None,
        animate = False
    ):

    if use_random_seed:
        board_np = get_board_seed(seed=init_state, size=size)
    else:
        board_np = get_board_int(n=init_state, size=size)


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

    # get cycle (list of boards) and period
    automata.advance(iterations=iterations)
    fist_board_cycle = automata.get_board_numpy(change_to_bool=True)
    board_cycle = [fist_board_cycle]
    while True:
        automata.advance() # next generation
        next_board = automata.get_board_numpy(change_to_bool=True)
        for i, b_iter in enumerate(board_cycle):
            if np.all(next_board == b_iter):
                if i>0:
                    board_cycle = board_cycle[i:]
                if animate:
                    print('Min alive cells:', get_min_on_cells(board_cycle))

                    # print patterns
                    print_patterns(board_cycle, all=False)

                    # identify pattern
                    identify_pattern(board_cycle)

                    automata.animate(interval=500)
                return board_cycle, len(board_cycle)
        else:
            board_cycle.append(next_board)

def run_cucles_analysis(
        size = 4,
        iterations = 100,
        sample_size = None,
        torus = True,
        use_fft = False,
        torch_device = None,
    ):

    assert np.log2(size) == int(np.log2(size)), 'size must be a power of 2'

    # cycle_counter = Counter()
    cycle_counter = defaultdict(list)

    total_cells = size * size

    use_random_seed = sample_size is not None

    tot_configurations = 2 ** total_cells

    if use_random_seed:
        # sample seeds from sample_size
        tot_states = sample_size
    else:
        # sample all
        tot_states = tot_configurations

    for init_state in tqdm(range(tot_states)):
        cycle, period = get_board_cycle_period(
            size = size,
            init_state = init_state,
            use_random_seed = use_random_seed,
            iterations = iterations,
            torus = torus,
            use_fft = use_fft,
            torch_device = torch_device,
        )

        cycle_counter[period].append(init_state)

    # print summary
    sample_str = \
        f'sample: {sample_size} ({tot_configurations} configurations)' \
        if use_random_seed \
        else f'exhaustive search ({tot_configurations} configurations)'
    print(f'size={size} ({size}x{size}={total_cells}) - {sample_str}')

    # sort it by period (biggest first)
    sorted_cycle_counts = sorted(cycle_counter.items(), key=lambda x: -x[0])
    for period, states in sorted_cycle_counts:
        states_extract = states if len(states)<10 else states[:4] + ['...'] + states[-4:]
        print(f'Period: {period}, States ({len(states)}): {states_extract}')

def test_get_board(n=15):
    a = get_board_int(n=15, size=2) # size=2 means 2x2 = 4 cells matrix with 16 configurations
    print(a)

def generate_analysis(size=2):

    use_random_seed = size not in [2,4]

    if use_random_seed:
        # sample `sample_size` states for size > 4 [8,16,...]
        run_cucles_analysis(
            size = size,
            iterations = 100,
            sample_size = 10000
        )
    else:
        # exaustive analyses for size in [2,4]
        run_cucles_analysis(
            size = size,
            iterations = 100,
        )

if __name__ == "__main__":

    size = 8

    # test_get_board(size)

    # generate_analysis(size)

    # visualize cycle animation for specific size and init state
    get_board_cycle_period(
        size = size,
        init_state = 27,
        use_random_seed = size not in [2,4],
        animate=True
    )


    '''
    size=8 (8x8=64) - sample: 1000 (18446744073709551616 configurations)
        Period: 132, States (89): [177, 197, 739, 881, '...', 9782, 9808, 9905, 9926]
        Period: 48, States (210): [5, 7, 32, 87, '...', 9804, 9820, 9830, 9988]
        Period: 32, States (166): [27, 93, 120, 132, '...', 9750, 9762, 9781, 9853]
        Period: 16, States (1): [2833]
        Period: 9, States (16): [1219, 1479, 1799, 2292, '...', 7876, 8487, 8578, 9694]
        Period: 8, States (6): [365, 3172, 5393, 6453, 8450, 9243]
        Period: 6, States (136): [69, 273, 355, 387, '...', 9521, 9593, 9857, 9858]
        Period: 2, States (761): [10, 24, 28, 53, '...', 9934, 9955, 9962, 9975]
        Period: 1, States (8615): [0, 1, 2, 3, '...', 9996, 9997, 9998, 9999]
    ---
    size=4 (4x4=16) - exhaustive search (65536 configurations)
        Period: 8, States (5248): [87, 93, 117, 171, '...', 64624, 64688, 64720, 64736]
        Period: 4, States (64): [63, 111, 159, 207, '...', 62208, 62976, 63744, 64512]
        Period: 2, States (3896): [7, 11, 13, 14, '...', 65156, 65168, 65216, 65280]
        Period: 1, States (56328): [0, 1, 2, 3, '...', 65532, 65533, 65534, 65535]
    ---
    size=2 (2x2=4) - exhaustive search (16 configurations)
        Period: 1, States (16): [0, 1, 2, 3, '...', 11, 12, 13, 15]
    '''

