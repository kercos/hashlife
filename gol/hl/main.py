import time
from gol.hl.lifeparsers import autoguess_life_file
from gol.hl.hashlife import (
    construct, ffwd, successor, join,
    expand, advance, centre
)
from gol.hl.render import render_img
import matplotlib.pyplot as plt
import os

# run the breeder forward many generations
def load_lif(fname):
    pat, comments = autoguess_life_file(fname)
    return construct(pat)

def ffwd_log(inputfile):
    init_t = time.perf_counter()
    pat = load_lif(inputfile)
    print(ffwd(pat, 64))
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
    pat = load_lif(inputfile)
    render_img(expand(pat))
    plt.savefig(f'{outputdir}/{filename}_0.png', bbox_inches='tight')
    render_img(expand(advance(centre(centre(pat)), 30)))
    plt.savefig(f'{outputdir}/{filename}_30.png', bbox_inches='tight')

    render_img(expand(advance(centre(centre(pat)), 120), level=0))
    plt.savefig(f'{outputdir}/{filename}_120_0.png', bbox_inches='tight')
    render_img(expand(advance(centre(centre(pat)), 120), level=1))
    plt.savefig(f'{outputdir}/{filename}_120_1.png', bbox_inches='tight')
    render_img(expand(advance(centre(centre(pat)), 120), level=2))
    plt.savefig(f'{outputdir}/{filename}_120_2.png', bbox_inches='tight')
    render_img(expand(advance(centre(centre(pat)), 120), level=3))
    plt.savefig(f'{outputdir}/{filename}_120_3.png', bbox_inches='tight')

if __name__ == '__main__':
    ffwd_log('input/hl_lifep/breeder.lif')
    expand_routine('input/hl_lifep/gun30.LIF')

