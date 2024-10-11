from gol.main import main_gol

##############
# BENCHMARKS
#
# Benchmark (conv2D): 1024x1024 grid, 1000 iters, torus=True
# Numpy (M1):                   31 Hz
# Torch mps (M1):              302 Hz
# Torch cuda (RTX 3090 Ti):   2866 Hz
#
# Benchmark (fft): 1024x1024 grid, 1000 iters, torus=True
# (fft is less efficient but you get torus for free)
# Numpy (M1):                   24 Hz
# Torch mps (M1):              217 Hz
# Torch cuda (RTX 3090 Ti):   2459 Hz
#
##############

def benchmark():
    # CONWAY GAME OF LIFE
    main_gol(
        shape_x = 2**10, # 2**10 == 1024,
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
        use_fft = False, # conv2d (more efficient)
        # torch_device = 'cpu', # torch cpu
        # torch_device = 'cuda', # torch cuda
        torch_device = 'mps', # torch mps
        # torch_device = 'npu', # torch npu # TODO: torch_device (npu) not recognized
        # torch_device = None, # numpy
    )

if __name__ == "__main__":
    benchmark()
