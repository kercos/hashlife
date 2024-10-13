import time
import numpy as np
import matplotlib.pyplot as plt
from gol.pure.automata import Automata
from gol.utils import (
    init_gol_board_neighborhood_rule,
    render_pure_img,
    render_pure_animation,
    numpy_to_life_106
)
from gol.hl.hashlife import (
    construct, ffwd, successor, join,
    expand, advance, centre, inner,
    render_img
)

outputdir = 'output/base'

def generate_hl_base(shape_x, file_life106=None):
    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        shape_x = shape_x,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
    )

    if file_life106 is not None:
        numpy_to_life_106(board, file_life106)

    # generate tuples (cells x,y coordinate which are 'on')
    pat_tuples = tuple(
        (x,y)
        for x in range(shape_x)
        for y in range(shape_x)
        if board[y,x]
    )

    # construct pattern
    init_t = time.process_time()
    node = construct(pat_tuples)
    t = time.process_time() - init_t
    print(f'Computation (hl-construct) took {t*1000.0:.1f} ms')

    return node, board, neighborhood, rule

def compute_hl_ffwd(node, giant_leaps, log=True):

    init_t = time.process_time()
    node, gens = ffwd(node, giant_leaps)
    t = time.process_time() - init_t
    print(f'Computation (hl-ffwd) took {t*1000.0:.1f} ms')

    if log:
        # print node info (k, X x Y, population, ...)
        print('node:', node)
        print('gens:', gens)
        print('successor:', successor.cache_info())
        print('join:', join.cache_info())

    return node, gens

def compute_hl_advance(node, iterations, log=True):

    init_t = time.process_time()
    node = advance(node, iterations)
    t = time.process_time() - init_t

    print(f'Computation (hl-advance) took {t*1000.0:.1f} ms')

    if log:
        # print node info (k, X x Y, population)
        print('node:', node)
        print('successor:', successor.cache_info())
        print('join:', join.cache_info())

    return node

def render_hl(node, filename, offset=None, show=True):
    # newnode = expand(advance(centre(centre(node)), gen), level=0)
    filepath = f'{outputdir}/{filename}.png'
    pts_hl = render_img(
        node,
        crop=False,
        offset=offset,
        name=filename, filepath=filepath,
        show=show, force_show=False
    )
    print('--> `hl` img:', filepath)
    return pts_hl

def main(
        shape_x = 16,
        giant_leaps = 1,
        render = False,
        animate = False,
        torch_device = None,
        log = True):

    '''Make sure hl-ffwd and hl-advance are considtent with pure'''

    filename = f'base{shape_x}'
    base_life106_filepath = f'{outputdir}/{filename}.LIFE'
    node, board, neighborhood, rule = generate_hl_base(shape_x, base_life106_filepath)

    if render:
        show_first = False # True if you want to show the first gen
        render_pure_img(
            board, neighborhood, rule,
            iterations=0,
            padding=None,
            filepath=f'{outputdir}/{filename}_0_pure.png',
            show=show_first,
            torch_device=torch_device
        )
        render_hl(node, f'{filename}_0_hl', offset=None, show=show_first)
        if show_first:
            plt.show() # show both

    # hl-ffwd
    assert giant_leaps is not None
    print(f'base {shape_x} ffwd')
    node_ffwd, gens = compute_hl_ffwd(node, giant_leaps, log=log)
    iterations = gens # get the generations equivalent to the giant leaps
    # hl-advance
    print(f'base {shape_x} advance')
    node_advance = compute_hl_advance(node, iterations, log=log)

    if node_advance.k != node_ffwd.k:
        # TODO: ask why this is the case
        print('<info> Reducing k for advance with `inner`')
        node_advance = inner(node_advance)

    assert node_ffwd.equals(node_advance)
    node = node_ffwd # same as node_advance

    if render or animate:
        # prepare padding for pure rendering
        new_size_pure = 2 ** node.k # sometime k-1 is ok but not always
        padding = (new_size_pure - shape_x) // 2 # before/after
        print('node k:', node.k, 'padding:', padding, 'new_size:', new_size_pure)

        if render:
            automata = render_pure_img(
                board, neighborhood, rule,
                iterations=iterations,
                filepath=f'{outputdir}/{filename}_{iterations}_pure.png',
                padding=padding,
                show=True,
                torch_device=torch_device
            )
            board_pure = automata.board
            assert new_size_pure  == board_pure.shape[0]

            # making offset in hl (by default it print on top-left corner as tight as possible)
            # get coordinates of `on` cells in board_pure
            pure_boord_on_cells_xy = np.array(tuple(zip(*np.where(board_pure==1))))
            upper_left_on_cell_xy = np.min(pure_boord_on_cells_xy,axis=0)
            board_hl = render_hl(
                node,
                filename=f'{filename}_{iterations}_hl',
                offset=upper_left_on_cell_xy,
                show=True
            )
            assert np.all(board_pure==board_hl)
            plt.show() # show both

        if animate:
            render_pure_animation(
                board, neighborhood, rule,
                iterations,
                padding=padding,
                name=f'{filename} 0 - {iterations}',
                interval_ms=0,
                torch_device=torch_device
            )

if __name__ == "__main__":

    # NOTE: hl gets stuck for shape_x=1024 in successor

    # make sure hl-ffwd and hl-advance are considtent with pure
    main(
        shape_x=128,
        giant_leaps = 2, # ffwd -> advance
        render=True,
        animate=False,
        torch_device = 'mps', # use Numpy if None
        log=True
    )

