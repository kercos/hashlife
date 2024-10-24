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
    '''
    Get board of given size for binary represetnation of n
    '''
    total_size = size * size
    bin_str = np.binary_repr(n, width=total_size)
    a = [bool(int(bit)) for bit in bin_str]
    return np.reshape(a, (size, size))

def get_board_seed(seed=123, size=4, density=0.5):
    '''
    Get board for given seed, size and density
    '''
    shape = (size, size)
    rng = np.random.default_rng(seed)
    board = rng.uniform(0, 1, shape)
    board = board < density
    return board

def get_min_on_cells(board_cycle):
    '''
    Get minimum alive cells in either element of the list of boards in board_cycle
    '''
    counts_on_cells = [
        np.count_nonzero(board_np)
        for board_np in board_cycle
    ]

    return np.min(counts_on_cells)

def print_patterns(board_cycle, all=False, pattern_filepath=None):
    '''
    Print min size (default) or all star pattern(s) in board_cycle (list of boards)
    If pattern_filepath is provided save the min size pattern
    '''
    if all:
        for c,b in enumerate(board_cycle, start=1):
            pattern_str = '\n'.join(numpy_to_stars(b, crop=True))
            print(f'-{c}-')
            print(pattern_str)
            print()

    all_pattern_str = []
    for c,b in enumerate(board_cycle, start=1):
        pattern_str = '\n'.join(numpy_to_stars(b, crop=True))
        all_pattern_str.append(pattern_str)
    len_pattern_str = [(len(p),p) for p in all_pattern_str]
    min_len_pattern_str = sorted(len_pattern_str, key=lambda lp: lp[0])[0] # get smallest
    pattern_str = min_len_pattern_str[1] # get pattern from pair
    if not all:
        print(f'---')
        print(pattern_str)
        print()
    if pattern_filepath is not None:
        with open(pattern_filepath, 'w') as fout:
            fout.write(pattern_str)
    return pattern_str

def identify_pattern(board_cycle):
    '''
    Try to identify patterns from live lexicon
    #TODO: update this to take into consideration https://conwaylife.com/wiki
    '''
    for b in board_cycle:
        pattern_str = '\n'.join(numpy_to_stars(b, crop=True))
        if pattern_str in PATTERN_NAME:
            name = PATTERN_NAME[pattern_str]
            print('FOUND PATTERN:', name)
            return name
    return None

def get_board_cycle_period(
        size = 2,
        rule = None,
        init_state = 7, # int (binary representation of matrix) or seed (random matrix)
        use_random_seed = False,
        jump_to_generation = 100,
        torus=True,
        use_fft = False,
        torch_device = None,
        min_cycle_period_to_report = 0,
        print_all_patterns = False,
        identify = False,
        force_animation = False,
        animate_if_new = False,
        save_to_file_if_new = False
    ):
    '''
    Analyze period of given board
    '''

    if use_random_seed:
        board_np = get_board_seed(seed=init_state, size=size)
    else:
        board_np = get_board_int(n=init_state, size=size)


    # init gol board and rule
    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        size = size,
        rule = rule,
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
    automata.advance(iterations=jump_to_generation)
    fist_board_cycle = automata.get_board_numpy(change_to_bool=True)
    board_cycle = [fist_board_cycle]
    while True:
        automata.advance() # next generation
        next_board = automata.get_board_numpy(change_to_bool=True)
        for i, b_iter in enumerate(board_cycle):
            if np.all(next_board == b_iter):
                if i>0:
                    board_cycle = board_cycle[i:]

                # identify pattern (False if identify is False)
                pattern_name = identify and identify_pattern(board_cycle)

                if force_animation or (animate_if_new and pattern_name):
                    cycle_period = len(board_cycle)
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                    print('FOUND NEW PATTERN CYCLE!')
                    print('Cycle period:', cycle_period)
                    print('Min alive cells:', get_min_on_cells(board_cycle))

                    if min_cycle_period_to_report is not None and cycle_period >= min_cycle_period_to_report:
                        if save_to_file_if_new:
                            pattern_filepath = f'output/cycles/size{size}_seed{init_state}.LIFS'
                        else:
                            pattern_filepath = None

                        # print patterns
                        print_patterns(
                            board_cycle,
                            all = print_all_patterns,
                            pattern_filepath = pattern_filepath
                        )

                        automata.animate(interval=500)
                return board_cycle, len(board_cycle)
        else:
            board_cycle.append(next_board)

def run_cycles_analysis(
        size = 4,
        rule = None,
        jump_to_generation = 100,
        sample_size = None,
        torus = True,
        use_fft = False,
        torch_device = None,
    ):
    '''
    Used for generate_analysis()
    '''

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
            rule = rule,
            init_state = init_state,
            use_random_seed = use_random_seed,
            jump_to_generation = jump_to_generation,
            torus = torus,
            min_cycle_period_to_report = None,
            identify = False,
            animate_if_new = False,
            save_to_file_if_new = False,
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

def generate_analysis(
        size = 2,
        rule = None,
        torch_device = None
    ):
    '''
    Generate exaustive (or partial) analysis
    '''
    use_random_seed = size not in [2,4]

    if use_random_seed:
        # sample `sample_size` states for size > 4 [8,16,...]
        run_cycles_analysis(
            size = size,
            rule = rule,
            jump_to_generation = 100,
            sample_size = 10000,
            torch_device = torch_device
        )
    else:
        # exaustive analyses for size in [2,4]
        run_cycles_analysis(
            size = size,
            rule = rule,
            jump_to_generation = 100,
            torch_device = torch_device
        )

def identify_patterns(
        size = 8,
        rule = None,
        min_cycle_period_to_report=132,
        iters = 1000,
        torch_device = None
    ):

    for _ in tqdm(range(iters)):

        init_state = np.random.randint(np.iinfo(np.int32).max)

        cycle, period = get_board_cycle_period(
            size = size,
            rule = rule,
            init_state = init_state,
            use_random_seed = True,
            jump_to_generation = 100,
            torus = True,
            use_fft = False,
            torch_device = torch_device,
            identify = True,
            min_cycle_period_to_report = min_cycle_period_to_report,
            animate_if_new = True,
            save_to_file_if_new = True
        )

def visualize_init_state(
        size = 8,
        rule = None,
        init_state = 27,
        torch_device = None
    ):
    get_board_cycle_period(
        size = size,
        rule = rule,
        init_state = init_state,
        use_random_seed = size not in [2,4],
        force_animation = True,
        print_all_patterns = False,
        torch_device = torch_device
    )

if __name__ == "__main__":

    '''test a random board'''
    # test_get_board(size)

    size = 8

    rule = None # default is GoL
    # rule = [[2, 3],[3, 6]] # HighLife

    torch_device = None # None for numpy (otherwise 'mps' or 'torch')

    '''
    Get compact analysis of periods cycles for given size
    (See printout below for size 2, 4, 8)
    '''
    # generate_analysis(
    #     size = size,
    #     rule = rule,
    #     torch_device = torch_device
    # )

    '''visualize cycle animation for specific size and init state'''
    visualize_init_state(
        size = size,
        rule = rule,
        init_state = 2833, # size must be 8 for init_state = 2833
        torch_device = torch_device
    )

    '''identify interesting patterns starting with board of given size'''
    # identify_patterns(
    #     size = size,
    #     rule = rule,
    #     min_cycle_period_to_report = 3,
    #     iters = 100,
    #     torch_device = torch_device
    # )

    ''' Printout of generate_analysis for some sizes (8, 4, 2)
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

    '''
    HighLife
    size=8 (8x8=64) - sample: 10000 (18446744073709551616 configurations)
        Period: 52, States (2): [6243, 7478]
        Period: 32, States (154): [20, 116, 190, 196, '...', 9705, 9713, 9724, 9833]
        Period: 12, States (1): [5414]
        Period: 9, States (15): [30, 1575, 2298, 2862, '...', 5517, 6845, 8141, 9834]
        Period: 8, States (84): [249, 270, 322, 652, '...', 9660, 9695, 9820, 9982]
        Period: 6, States (14): [556, 1230, 1560, 3635, '...', 7182, 7837, 8634, 8682]
        Period: 4, States (12): [514, 3756, 4446, 5151, '...', 7390, 8166, 8731, 9945]
        Period: 2, States (707): [19, 23, 28, 41, '...', 9933, 9976, 9991, 9994]
        Period: 1, States (9011): [0, 1, 2, 3, '...', 9996, 9997, 9998, 9999]
    '''

