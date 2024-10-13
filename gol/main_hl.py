import time
from gol.hl.lifeparsers import autoguess_life_file
from gol.hl.hashlife import (
    construct, ffwd, successor, join,
    expand, advance, centre, render_img
)
import matplotlib.pyplot as plt
import os

# run the breeder forward many generations
def load_lif(fname):
    pat_tuples, comments = autoguess_life_file(fname)
    return construct(pat_tuples)

def ffwd_log(inputfile):
    init_t = time.perf_counter()
    node = load_lif(inputfile)
    print(ffwd(node, 64))
    t = time.perf_counter() - init_t
    print(f'Computation took {t*1000.0:.1f}ms')
    print(successor.cache_info())
    print(join.cache_info())


def expand_routine(inputfile):
    filename_ext = os.path.basename(inputfile)
    filename, ext = os.path.splitext(filename_ext)
    os.path.splitext(filename_ext)
    outputdir = 'output/hl_imgs'

    ## test the Gosper glider gun
    # 00 generations (level=0)
    gen = 0
    level = 0
    node = load_lif(inputfile)
    filename_gen_level = f'{filename}_{gen}_{level}'
    filepath = f'{outputdir}/{filename_gen_level}.png'
    render_img(node, name=filename_gen_level, filepath=filepath)

    # 30 generations (level=0)
    gen = 30
    level = 0
    node_30 = advance(centre(centre(node)), gen)
    filename_gen_level = f'{filename}_{gen}_{level}'
    filepath = f'{outputdir}/{filename_gen_level}.png'
    render_img(node_30, level=level, name=filename_gen_level, filepath=filepath)

    # 120 generations (different levels)
    gen = 120
    node_120 = advance(centre(centre(node)), gen)
    for level in [0,1,2,3]:
        filename_gen_level = f'{filename}_{gen}_{level}'
        filepath = f'{outputdir}/{filename_gen_level}.png'
        render_img(node_120, level=level, name=filename_gen_level, filepath=filepath)

if __name__ == '__main__':
    ffwd_log('input/hl_lifep/breeder.lif')
    expand_routine('input/hl_lifep/gun30.LIF')

