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

def get_base16_board_nb_rule():
    board, neighborhood, rule = init_gol_board_nb_rule(
        shape_x = 16,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
    )
    return board, neighborhood, rule

def make_base_file(filepath):
    board, neighborhood, rule = get_base16_board_nb_rule()
    filepath = filepath
    numpy_to_life_106(board, filepath)

def test_ffw(filepath):
    pat_tuples, comments = autoguess_life_file(filepath)
    pat = construct(pat_tuples)
    init_t = time.perf_counter()
    print(ffwd(pat, 1000))
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

def test_animate(expand = None, interval_ms=0):
    from gol.pure.main import Automata
    import numpy as np
    board, neighborhood, rule = get_base16_board_nb_rule()
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

if __name__ == "__main__":
    base16_filepath = 'output/base16.LIFE'
    # make_base_file(base16_filepath)
    # test_ffw(base16_filepath)
    # test_render(base16_filepath)
    # test_animate()
    # test_animate(expand = 20, interval_ms=0)