import os
import numpy as np
from gol.pure.automata import Automata
from matplotlib import pyplot as plt

def init_gol_board_neighborhood_rule(
        size = 16,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
    ):

    shape = (size, size)

    if type(initial_state) is str:
        if initial_state == 'random':
            # initialize random generator
            rng = np.random.default_rng(seed)
            board = rng.uniform(0, 1, shape)
            board = board < density
        elif initial_state == 'square':
            sq = 2 # alive square size in the middle of the board
            assert sq % 2 == 0
            board = np.zeros(shape)
            board[
                size//2-sq//2:size//2+sq//2,
                size//2-sq//2:size//2+sq//2
            ] = 1 # alive
        else:
            assert initial_state.endswith('.npy'), 'wrong initial_state (must end with .npy)'
            board = np.load(initial_state)
    elif type(initial_state) is np.ndarray:
        board = initial_state
    else:
        assert(False), 'wrong initial_state arg'

    neighborhood = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
    )

    # GoL Rule (same as Conway class):
    rule = [
        [2, 3], # 'on->on': (2,3): "on" neighbours (can't contain 0)
        [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
    ]

    # GoL Rule with neighborhood all ones:
    # neighborhood = np.ones((3,3))
    # rule = [
    #     [3, 4], # 'on->on': (2,3): "on" neighbours (can't contain 0)
    #     [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
    # ]

    # exploring other rules:
    # rule = [
    #     [2], # 'on->on': "on" neighbours (can't contain 0)
    #     [1]  # 'off->on':   "on" neighbours (can't contain 0)
    # ]

    return board, neighborhood, rule

'''
Returns a list of
- (x,y) points (only_alive=True, default)
- (x,y, gray) points (only_alive=False),
    where gray==0. for off and gray==1. for on cells
'''
def get_board_pts(board_np, only_alive=True):
    size = board_np.shape[0]
    if only_alive:
        return [
            (x,y)
            for x in range(size)
            for y in range(size)
            if board_np[y,x]
        ]
    else:
        return [
            (
                x,
                y,
                1. if board_np[y,x] else 0.
            )
            for x in range(size)
            for y in range(size)
        ]

def show_board_np(board, name, force_show=True):
    plt.figure(name, figsize=(5, 5))
    plt.imshow(
        board,
        interpolation="nearest",
        cmap=plt.cm.gray
    )
    plt.axis("off")

    if force_show:
        plt.show()

def render_pure_img(
        board, neighborhood, rule,
        iterations=0, padding = None,
        filepath=None, show=True,
        torch_device=None):

    # adjust padding
    if padding:
        pad_before_after = padding
        board = np.pad(board, pad_before_after)

    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = False,
        use_fft = False,
        torch_device = torch_device, # numpy
    )
    if iterations > 0:
        automata.benchmark(iterations)

    if filepath:
        automata.save_last_frame(filepath)

    if show:
        name = os.path.splitext(os.path.basename(filepath))[0]
        automata.show_current_frame(name, force_show=False)
        # need to trigger plt.show() outside

    print('--> `pure` img:', filepath)
    return automata

def render_pure_animation(
        board, neighborhood, rule,
        iterations,
        padding = None,
        name = 'animation',
        interval_ms=0,
        torch_device=None):

    # adjust padding
    if padding:
        pad_before_after = padding
        board = np.pad(board, pad_before_after)

    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = True,
        use_fft = False,
        torch_device = torch_device,
    )
    automata.animate(name=name, iterations=iterations, interval=interval_ms)

'''
Convert board to . and * string pattern (life 1.05)
'''
def numpy_to_stars(board_np, crop=False):
    if crop:
        nonzero = np.nonzero(board_np)
        if len(nonzero[0])>0:
            min_y = np.min(nonzero[0])
            max_y = np.max(nonzero[0])
        else:
            min_y = 0
            max_y = board_np.shape[0]-1
        if len(nonzero[1])>0:
            min_x = np.min(nonzero[1])
            max_x = np.max(nonzero[1])
        else:
            min_x = 0
            max_x = board_np.shape[1]-1
        board_np = board_np[min_y:max_y+1, min_x:max_x+1]
    return [
        ''.join('*' if c else '.' for c in line)
        for line in board_np.tolist()
    ]

'''
Convert board to . and * file format (life 1.05)
'''
def numpy_to_life_105(board_np, filepath):
    # see http://www.mirekw.com/ca/ca_files_formats.html
    pattern_lines = numpy_to_stars(board_np)
    pattern_lines = '\n'.join(pattern_lines)
    with open(filepath, 'w') as fout:
        fout.writelines(pattern_lines)



'''
Convert a (dense) NumPy array to list of (x,y) positions in life 1.06
'''
def numpy_to_life_106(board_np, filepath):
    # see http://www.mirekw.com/ca/ca_files_formats.html
    header = '#Life 1.06'
    size, shape_y = board_np.shape

    lines = [header]
    for x in range(size): # columns (x)
        for y in range(shape_y): #rows (y)
            if board_np[y,x]:
                lines.append(f'{x} {y}') # x,y

    # write to file
    with open(filepath, 'w') as fout:
        # lines with return except for last
        lines_with_return = \
            [f'{l}\n' for l in lines[:-1]] + \
            [lines[-1]]
        fout.writelines(lines_with_return)

'''
Convert board to RLE
'''
def numpy_to_rle(board_np, filepath):
    # see http://www.mirekw.com/ca/ca_files_formats.html
    from gol.hl.lifeparsers import write_rle

    pts = get_board_pts(board_np, only_alive=True)
    size = board_np.shape[0]
    write_rle(fixed_size=size, filepath=filepath, pts=pts)


if __name__ == "__main__":
    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        size = 16,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
    )
    filepath_out_life = 'output/base16_0.LIFE'
    filepath_out_rle = 'output/base16_0.RLE'
    numpy_to_life_106(board, filepath_out_life)
    numpy_to_rle(board, filepath_out_rle)