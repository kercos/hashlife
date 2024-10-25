from gol.main_pure import main_gol

##############
# BENCHMARKS (Pure) 1024x1024 grid, 1000 iters, torus=True
#
# Conv2D:
# - Numpy (M1):                   31 Hz
# - Torch mps (M1):              302 Hz (? --> 199 Hz)
# - Torch cuda (RTX 3090 Ti):   2866 Hz
#
# Conv2D-POLY:
# - Numpy (M1):                   32 Hz
# - Torch mps (M1):              369 Hz
# - Torch cuda (RTX 3090 Ti):   ____ Hz
#
# FFT
# (fft is less efficient but you get torus for free)
# - Numpy (M1):                   24 Hz
# - Torch mps (M1):              217 Hz
# - Torch cuda (RTX 3090 Ti):   2459 Hz
#
# FFT-POLY
# - Numpy (M1):                   24 Hz
# - Torch mps (M1):              355 Hz
# - Torch cuda (RTX 3090 Ti):   ____ Hz
#
# BENCHMARKS (Golly) 16x16 grid, 1000 iters, torus=True
#
# - Size 16 Iters 1000 cells in 0.0 s 261097 Hz (board) 0.07 BHz (cell)
# TODO: Fatal error: Illegal whitespace after count (for board > 16)
##############

def benchmark_golly():
    from gol.test_golly import generate_base, benchmark_golly, run_golly
    from gol.utils import numpy_to_rle

    # TODO: Fatal error: Illegal whitespace after count (for board > 16)
    size = 16 # 2**10 # 2**10 == 1024,
    iterations = 1000

    node, board, neighborhood, rule = generate_base(
        size = size,
        seed = 123
    )

    rle_filepath_zero = f'output/golly/base{size}_0.RLE'
    numpy_to_rle(board, rle_filepath_zero)

    rle_filepath_golly = f'output/golly/base{size}_{iterations}_golly.RLE'
    run_golly(
        rle_filepath = rle_filepath_zero,
        iterations = iterations,
        output_path = rle_filepath_golly
    )
    benchmark_golly(
        size = size,
        rle_filepath = rle_filepath_zero,
        iterations = iterations
    )


def benchmark_pure():
    # CONWAY GAME OF LIFE
    # see Automata.benchmark
    main_gol(
        size = 2**10, # 2**10 == 1024,
        initial_state = 'random', # 'square', 'filenmae.npy'
        density = 0.5, # only used with initial_state=='random'
        seed = 123, # only used with initial_state=='random'
        iterations = 1000,
        torus = True,
            # - fft (numpy, torch) always True TODO: fix me
            # - conv2d
            #   - numpy: works :)
            #   - torch: works :)
        animate = False, # benchmark if False
        show_last_frame = False, # only applicable for benchmark
        save_last_frame = None, # 'test.png' '100k.npy'
        use_fft = True, # conv2d (more efficient)
        use_poly_update = True,
        # torch_device = 'cpu', # torch cpu
        # torch_device = 'cuda', # torch cuda
        torch_device = 'mps', # torch mps
        # torch_device = 'npu', # torch npu # TODO: torch_device (npu) not recognized
        # torch_device = None, # numpy
    )

if __name__ == "__main__":
    benchmark_pure()
    # benchmark_golly()
