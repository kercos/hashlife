import os
import glob
import time
import math
import random
from gol.base import generate_base
from gol.hl.hashlife import (
    construct, ffwd, successor, join, crop,
    expand, advance, centre, inner,
    render_img
)

outputdir = 'output/jumps'

def compute_hl_ffwd(node, giant_leaps, timer=False, log=True):

    if timer:
        init_t = time.process_time()
        node, gens = ffwd(node, giant_leaps)
        t = time.process_time() - init_t
        print(f'Computation (hl-ffwd) took {t*1000.0:.1f} ms')
    else:
        node, gens = ffwd(node, giant_leaps)

    if log:
        # print node info (k, X x Y, population, ...)
        print('node:', node)
        print('gens:', gens)
        print('successor:', successor.cache_info())
        print('join:', join.cache_info())

    return node, gens

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


def test_hl_pruning(size, seed, log=False):

    basefilename = f'base{size}_s{seed}'
    base_life106_filepath = f'{outputdir}/{basefilename}.LIFE'
    node, board, neighborhood, rule = generate_base(
        size=size,
        seed=seed,
        file_life106=base_life106_filepath
    )

    crop = True

    # final_size = 32
    # pad_before_after = (final_size - size) // 2
    # board_padding = np.pad(board, pad_before_after)

    # automata = Automata(
    #     board = board_padding,
    #     neighborhood = neighborhood,
    #     rule = rule,
    #     torus = False,
    #     use_fft = False,
    #     torch_device = None, # numpy
    # )

    if log:
        print(f'base{size}', node)

    giant_leaps_per_jumpt = 2

    max_jump = 100
    max_jump_zceil = math.ceil(math.log10(max_jump))

    max_gl = max_jump * giant_leaps_per_jumpt
    max_gl_zceil = math.ceil(math.log10(max_gl))

    gl = 0
    i = 0

    for j in range(1,max_jump+1):

        if log:
            print('----------------')
            print('j:', j)

        j_zf = f'{j}'.zfill(max_jump_zceil)

        gl += giant_leaps_per_jumpt
        gl_zf = f'{gl}'.zfill(max_gl_zceil)

        node_jump, gens = compute_hl_ffwd(
            node,
            giant_leaps = giant_leaps_per_jumpt,
            log = False
        )
        i += gens
        i_zf = f'{i}'.zfill(5)

        if log:
            print(f'jump={j} gl={gl}, i={i}')
            print('N:', node)
            print('A:', node_jump)

        # automata.benchmark(iterations=gens)

        # automata.show_current_frame(
        #     name = f'{basefilename}_{final_size}_pure_{gens}',
        #     force_show = False
        # )

        if log:
            filename = f'{basefilename}_j{j_zf}_gl{gl_zf}_i{i_zf}_A'
            filepath = f'{outputdir}/{filename}'
            render_img(
                node_jump,
                name = filename,
                filepath = filepath,
                crop = crop,
                show = False
            )

        node_jump_inner = inner(node_jump)
        if log:
            print('I:', node_jump_inner)

        if log:
            filename_inner = f'{basefilename}_j{j_zf}_gl{gl_zf}_i{i_zf}_I'
            filepath_inner = f'{outputdir}/{filename_inner}'
            render_img(
                node_jump_inner,
                name = filename_inner,
                filepath = filepath_inner,
                crop = crop,
                show = False
            )

        # show all
        # plt.show()


        jump_inner_same_as_node = expand(node) == expand(node_jump_inner)
        if log:
            print('Jump-inner same as node', jump_inner_same_as_node)

        if jump_inner_same_as_node:
            return j

        node = node_jump_inner


if __name__ == "__main__":

    size = 128
    max_j = -1

    for i in range(100):
        seed = random.randint(1,999999999)
        print('-- SEED', seed)

        last_j = test_hl_pruning(
            size=size,
            seed=seed,
            log=False
        )

        print('-- LAST JUMP', last_j)

        if last_j > max_j:

            # update max jump
            max_j = last_j

            # remove all files
            all_jump_files = os.listdir(outputdir)
            for f in all_jump_files:
                os.remove(f'{outputdir}/{f}')

            last_j = test_hl_pruning(
                size=size,
                seed=seed,
                log=True
            )

            # remove all files excluding current seed
            # all_jump_files = os.listdir(outputdir)
            # for f in all_jump_files:
            #     if not f.startswith(f'base{size}_s{seed}'):
            #         os.remove(f'{outputdir}/{f}')

