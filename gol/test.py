import os
import time
from gol.pure.main import init_gol_board_nb_rule
from gol.utils import numpy_to_life_106
from gol.hl.lifeparsers import autoguess_life_file
from gol.hl.hashlife import (
    construct, ffwd, successor, join,
    expand, advance, centre
)
from gol.hl.render import render_img
import matplotlib.pyplot as plt

def get_base_board_nb_rule(shape_x):
    board, neighborhood, rule = init_gol_board_nb_rule(
        shape_x = shape_x,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
    )
    return board, neighborhood, rule

def make_base_file(shape_x, filepath):
    board, neighborhood, rule = get_base_board_nb_rule(shape_x)
    filepath = filepath
    numpy_to_life_106(board, filepath)

def test_ffw(filepath, n=1000):
    pat_tuples, comments = autoguess_life_file(filepath)
    pat = construct(pat_tuples)
    init_t = time.perf_counter()
    print(ffwd(pat, n))
    t = time.perf_counter() - init_t
    print(f'Computation took {t*1000.0:.1f} ms')
    print(successor.cache_info())
    print(join.cache_info())

def test_advance(filepath, n=1000):
    pat_tuples, comments = autoguess_life_file(filepath)
    pat = construct(pat_tuples)
    init_t = time.perf_counter()
    print(advance(pat, n))
    t = time.perf_counter() - init_t
    print(f'Computation took {t*1000.0:.1f} ms')
    print(successor.cache_info())
    print(join.cache_info())

def test_render(filepath):
    outputdir = 'output'
    filename_ext = os.path.basename(filepath)
    filename, ext = os.path.splitext(filename_ext)
    pat_tuples, comments = autoguess_life_file(filepath)
    pat = construct(pat_tuples)
    init_t = time.perf_counter()
    for gen in [0,1000]:
        render_img(expand(advance(centre(centre(pat)), gen), level=0))
        plt.savefig(f'{outputdir}/{filename}_{gen}_0.png', bbox_inches='tight')
    t = time.perf_counter() - init_t
    print(f'Rendering took {t*1000.0:.1f} ms')
    print(successor.cache_info())
    print(join.cache_info())

def test_base_animate(shape_x, expand = None, interval_ms=0):
    from gol.pure.main import Automata
    import numpy as np
    board, neighborhood, rule = get_base_board_nb_rule(shape_x)
    if expand:
        pad_before_after = expand
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

def main_base16():
    shape_x = 16
    base16_filepath = 'output/base16.LIFE'
    make_base_file(shape_x, base16_filepath)

    print('test ffw')
    test_ffw(base16_filepath)
    print('test advance')
    test_advance(base16_filepath)

    test_render(base16_filepath)
    # test_base_animate(shape_x=16)
    test_base_animate(shape_x=16, expand=20, interval_ms=0)

def main_base1k():
    shape_x = 1024
    base1k_filepath = 'output/base1k.LIFE'
    make_base_file(shape_x, base1k_filepath)

    print('test ffw')
    test_ffw(base1k_filepath)
    # TODO: gets stuck in ffw (successor)

    # print('test advance')
    # test_advance(base1k_filepath, n=10)
    # TODO: gets stuck in advance (successor)

    # test_render(base1k_filepath)
    # test_base_animate(shape_x)
    # test_base_animate(shape_x=16, expand=20, interval_ms=0)

if __name__ == "__main__":
    main_base16()
    # main_base1k() # TODO: gets stuck in ffw/advance (successor)