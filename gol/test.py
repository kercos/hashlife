import time
import numpy as np
from gol.utils import init_gol_board_neighborhood_rule
from gol.utils import numpy_to_life_106
from gol.hl.hashlife import (
    construct, ffwd, successor, join,
    expand, advance, centre
)
from gol.hl.render import render_img
import matplotlib.pyplot as plt
from gol.pure.automata import Automata


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
    pat = construct(pat_tuples)

    return pat, board, neighborhood, rule

def test_ffw(pat, iterations, log=True):

    init_t = time.perf_counter()
    node = ffwd(pat, iterations)
    t = time.perf_counter() - init_t

    print(f'Computation (ffw) took {t*1000.0:.1f} ms')

    if log:
        # print node info (k, X x Y, population, ...)
        print(node)
        print(successor.cache_info())
        print(join.cache_info())

    return node

def test_advance(pat, iterations, log=True):

    init_t = time.perf_counter()
    node = advance(pat, iterations)
    t = time.perf_counter() - init_t

    print(f'Computation (advance) took {t*1000.0:.1f} ms')

    if log:
        # print node info (k, X x Y, population)
        print(node)
        print(successor.cache_info())
        print(join.cache_info())

    return node

def test_render_hl(pat, filename, iterations):
    outputdir = 'output/base'
    # also render for t=0
    for gen in [0, iterations]:
        render_img(expand(advance(centre(centre(pat)), gen), level=0))
        filepath = f'{outputdir}/{filename}_{gen}_0.png'
        plt.savefig(filepath, bbox_inches='tight')
        print('See `hl` img:', filepath)

def test_render_pure_img(shape_x, filepath, iterations, padding = None):
    # get base
    _, board, neighborhood, rule = generate_hl_base(shape_x)

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
        torch_device = 'mps', # numpy
    )
    automata.benchmark(iterations)
    automata.save_last_frame(filepath)

    print('See `pure` img:', filepath)


def test_render_pure_animate(shape_x, padding = None, interval_ms=0):
    # get base
    _, board, neighborhood, rule = generate_hl_base(shape_x)

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
        torch_device = None, # numpy
    )
    automata.animate(interval_ms) #ms

def main(
        shape_x=16,
        method='ffw',
        iterations = 1000,
        render=False,
        animate=False,
        log=True):

    assert method in ['ffw', 'advance'], \
        'method must be `ffw` or `advance`'

    filename = f'base{shape_x}'
    base_life106_filepath = f'output/base/{filename}.LIFE'
    pat, board, neighborhood, rule = generate_hl_base(shape_x, base_life106_filepath)

    if render:
        test_render_hl(pat, f'{filename}_{method}', iterations)


    if method == 'ffw':
        print(f'test {shape_x} ffw')
        node, gens = test_ffw(pat, iterations, log=log)
        # TODO: gets stuck for shape_x=1024 in ffw (successor)
    else:
        assert method == 'advance'
        print(f'test {shape_x} advance')
        node = test_advance(pat, iterations, log=log)
        # TODO: gets stuck for shape_x=1024 in ffw (successor)
        new_shape_x = 2 ** node.k
        padding = (new_shape_x - shape_x) // 2 # before/after
        print('node k:', node.k, 'padding:', padding, 'new_shape:', new_shape_x)
        if render:
            png_last = f'output/base/{filename}_{iterations}_pure.png'
            test_render_pure_img(
                shape_x,
                png_last,
                iterations=iterations,
                padding=padding
            )
        if animate:
            test_render_pure_animate(shape_x=shape_x, padding=padding, interval_ms=0)

if __name__ == "__main__":

    # works ok
    # for method in ['ffw','advance']:
    # for method in ['ffw']:
    for method in ['advance']:
        main(
            shape_x=128,
            method=method,
            iterations = 100,
            render=True,
            animate=True,
            log=False
        )

