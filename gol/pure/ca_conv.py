'''
this code is from:
https://gist.github.com/njbbaer/4da02e2960636d349e9bae7ae43c213c
but see also:
- julia: https://rivesunder.github.io/SortaSota/2021/09/27/faster_life_julia.html
- carle: https://github.com/rivesunder/carle
- https://github.com/moritztng/cellular
'''

import time
import numpy as np
from numpy.fft import fft2 as np_fft2, ifft2 as np_ifft2
from matplotlib import pyplot, animation
import torch
from torch.fft import fft2 as torch_fft2, ifft2 as torch_ifft2
import scipy

class Automata:
    '''
    shape: must be 2d and power of 2 to make things efficient
    board: initial configuration (binary)
    neighborhood: who are my neighbors (maked with 1s)
    torus: rolling over the boundaries (https://en.wikipedia.org/wiki/Torus)
    rule[0]: '1->1' based on "1" neighbours (can't contain 0)
    rule[1]: '0->1' based on "1" neighbours (can't contain 0)
    torch_device: None or in ["cpu", "cuda", "mps"]
    '''
    def __init__(
            self, board, neighborhood, rule, torus=True,
            use_fft = False, # otherwise conv2d
            torch_device = None,
    ):

        assert (
            0 not in rule[0]
            and
            0 not in rule[1]
        ), "Rule cannot contain zeros"

        assert (
            torch_device in [None, "cpu", "cuda", "mps"]
        ), f"torch_device ({torch_device}) not recognized"

        # board - the grid
        self.board = board

        # neighborhood (e.g, 3x3 in GoL)
        self.neighborhood = neighborhood

        self.shape_x, _, = self.height, self.width = self.shape = self.board.shape

        assert self.height == self.width

        self.minus_shape_x_half_plus_one =  - int(self.shape_x / 2) + 1

        self.rule = rule
        self.torus = torus

        # torch
        self.torch_device = torch_device
        self.use_torch = torch_device is not None

        if use_fft:
            nh, nw = self.neighborhood.shape # say (3,3) for Conway's GoL

            # create the FFT kernal (init as zero)
            # put neighborhood mask in the middle (everything else is 0.)
            # TODO check:
            # - not always perfectly centered
            # - e.g., with 3x3 neighborhood and grid size being even (power of two)
            self.kernal = np.zeros(self.shape, dtype=np.float32) # kernal has same shape of board, say (256,256)
            self.kernal[
                (self.shape[0] - nh - 1) // 2 : (self.shape[0] + nh) // 2,
                (self.shape[1] - nw - 1) // 2 : (self.shape[1] + nw) // 2
            ] = self.neighborhood

        if self.use_torch:
            # need to convert arrays to tensor when using torch
            # print('cuda available:', torch.cuda.is_available())
            # print('mps available:', torch.backends.mps.is_available())
            torch_device = torch.device(torch_device)
            torch.set_default_device(torch_device)

            if use_fft:
                # less efficient (buth worth trying)
                self.torch_conv_func = self.torch_conv_fft
                self.board = torch.from_numpy(self.board).to(torch_device)
                self.rule = [torch.IntTensor(r).to(torch_device) for r in self.rule] # int
                self.kernal = torch.from_numpy(self.kernal).to(torch_device)
                self.kernal_ft = torch_fft2(self.kernal)
                self.rule_dtype = torch.int32
            else:
                # use conv2d (more efficient) - TODO: check why only works on floats
                # need to convert each rule to float tensor
                self.torch_conv_func = self.torch_conv_conv2d
                self.board = torch.from_numpy(self.board).float().to(torch_device)
                self.rule = [torch.FloatTensor(r).to(torch_device) for r in self.rule] # float
                self.neighborhood = torch.from_numpy(self.neighborhood).float().to(torch_device)
                self.rule_dtype = torch.float32
        else:
            # numpy
            if use_fft:
                # less efficient (buth worth mentioning)
                self.numpy_conv_func = self.np_conv_fft
                self.kernal_ft = np_fft2(self.kernal) # same shape but floating numbers
            else:
                # use conv2d (more efficient)
                self.numpy_conv_func = self.np_conv_conv2d

    '''
    Main count_real operation in a single time-step
    Based on numpy and conv2d
    '''
    def np_conv_conv2d(self):

        # the conv2d step (via scipy)
        counts_int = scipy.signal.convolve2d(self.board, self.neighborhood, mode='same')

        # rolling over the boundaries
        # see https://en.wikipedia.org/wiki/Torus
        if self.torus:
            counts_int = self.np_roll_boundaries(counts_int)

        return counts_int

    '''
    Main count_real operation in a single time-step
    Based on numpy and fft
    '''
    def np_conv_fft(self):
        # fft2 same shape but floating numbers
        board_ft = np_fft2(self.board)

        # inverted fft2 (complex numbers)
        inverted = np_ifft2(board_ft * self.kernal_ft)

        # get the real part of the complex argument (real number)
        count_real = np.real(inverted)

        # round real part to closest integer
        counts_int = np.rint(count_real)

        # rolling over the boundaries
        # see https://en.wikipedia.org/wiki/Torus
        if self.torus:
            counts_int = self.np_roll_boundaries(counts_int)

        return counts_int

    '''
    Rolling over the boundaries (numpy version)
    see https://en.wikipedia.org/wiki/Torus
    '''
    def np_roll_boundaries(self, counts_int):
        return np.roll(
            counts_int,
            self.minus_shape_x_half_plus_one,
            axis=(0,1)
        )

    '''
    Main count_real operation in a single time-step (torch version)
    Usign FFT (not as efficient as torch.nn.functional.conv2d)
    '''
    def torch_conv_fft(self):
        # fft2 same shape but floating numbers
        board_ft = torch_fft2(self.board)

        # inverted fft2 (complex numbers)
        inverted = torch_ifft2(board_ft * self.kernal_ft)

        # get the real part of the complex argument (real number)
        count_real = torch.real(inverted)

        # round real part to closest integer
        counts_int = torch.round(count_real).int()

        if self.torus:
            # rolling over the boundaries
            counts_int = self.torch_roll_boundaries(counts_int)

        return counts_int

    '''
    Main count_real operation in a single time-step
    torch version using conv2d (more efficient)
    '''
    def torch_conv_conv2d(self):

        # create two more dims
        board_conv = self.board[None,None,:,:]
        nb_conv = self.neighborhood[None,None,:,:]

        # the conv2d step
        counts_int = torch.nn.functional.conv2d(board_conv, nb_conv, padding='same')

        # taking only first two channels in first two dim
        counts_int = counts_int[0,0,:,:]

        if self.torus:
            # rolling over the boundaries
            counts_int = self.torch_roll_boundaries(counts_int)

        return counts_int

    '''
    Main count_real operation in a single time-step
    torch version using fft (less efficient)
    '''
    def torch_conv_fft(self):

        # fft2 same shape but floating numbers
        board_ft = torch_fft2(self.board)

        # inverted fft2 (complex numbers)
        inverted = torch_ifft2(board_ft * self.kernal_ft)

        # get the real part of the complex argument (real number)
        count_real = torch.real(inverted)

        # round real part to closest integer
        counts_int = torch.round(count_real).int()

        if self.torus:
            # rolling over the boundaries
            counts_int = self.torch_roll_boundaries(counts_int)

        return counts_int

    '''
    Rolling over the boundaries (torch version)
    see https://en.wikipedia.org/wiki/Torus
    '''
    def torch_roll_boundaries(self, counts_int):
        return torch.roll(
            counts_int,
            (
                self.minus_shape_x_half_plus_one,
                self.minus_shape_x_half_plus_one
            ), # need tuple here because torch behaves differently from numpy apparently
            dims=(0,1)
        )

    '''
    Apply the rule (numpy version)
    '''
    def np_apply_rule(self, count_ones_neighbours):

        board_ones = self.board == 1
        board_zeros = ~ board_ones # negation

        new_board = np.zeros(self.shape)

        # rule[0] (survival): '1->1' based on count of "1" neighbours
        new_board[
            np.where(
                np.isin(count_ones_neighbours, self.rule[0]).reshape(self.shape)
                &
                board_ones # on cells
            )
        ] = 1

        # rule[1] (reproduction): '0->1' based on count of "1" neighbours
        new_board[
            np.where(
                np.isin(count_ones_neighbours, self.rule[1]).reshape(self.shape)
                &
                board_zeros # off cells
            )
        ] = 1

        # all other cells stay zeros (1->0, 0->0)

        self.board = new_board

    '''
    Apply the rule (torch version)
    '''
    def torch_apply_rule(self, count_ones_neighbours):
        board_ones = self.board == 1
        board_zeros = ~ board_ones # negation

        new_board = torch.zeros(self.shape, dtype=self.rule_dtype) # int or float

        # rule[0] (survival): '1->1' based on count of "1" neighbours
        new_board[
            torch.where(
                torch.isin(count_ones_neighbours, self.rule[0]).reshape(self.shape)
                &
                board_ones # on cells
            )
        ] = 1

        # rule[1] (reproduction): '0->1' based on count of "1" neighbours
        new_board[
            torch.where(
                torch.isin(count_ones_neighbours, self.rule[1]).reshape(self.shape)
                &
                board_zeros # off cells
            )
        ] = 1

        # all other cells stay zeros (1->0, 0->0)

        self.board = new_board

    '''
    Step update function using numpy
    '''
    def np_update_board(self):

        # counting number of alive cells in neighbourhood (same shape)
        # via fft or conv2 based on torch_conv_func
        # (np_conv_fft or np_conv_conv2d)
        count_ones_neighbours = self.numpy_conv_func()

        # apply rule (update board inplace)
        self.np_apply_rule(count_ones_neighbours)

    '''
    Step update function using torch
    '''
    def torch_update_board(self):

        # counting number of alive cells in neighbourhood (same shape)
        # via fft or conv2 based on torch_conv_func
        # (torch_conv_fft or torch_conv_conv2d)
        count_ones_neighbours = self.torch_conv_func()

        # apply rule (update board inplace)
        self.torch_apply_rule(count_ones_neighbours)

    '''
    Main Benchmark to run it as fast as possible (without visualizing it)
    '''
    def benchmark(self, iterations):
        start = time.process_time() # TODO: change to pytorch profiler

        if self.use_torch:
            for _ in range(iterations):
                self.torch_update_board()
        else:
            # use numpy
            for _ in range(iterations):
                self.np_update_board()

        ellapsed = time.process_time() - start # TODO: change to pytorch profiler

        hz = iterations / ellapsed

        hz_B_cell = hz * self.shape_x * self.shape_x / 10 ** 9 # Billions

        print(
            "Performed", iterations,
            "iterations of", self.shape,
            f"cells in {ellapsed} s",
            f"{hz:.0f} Hz (board)",
            f"{hz_B_cell:.2f} BHz (cell)"
        )

    '''
    Show grid in real time (rely on numpy for now)
    '''
    def animate(self, interval=100):

        def update_animation(*args):
            self.np_update_board()
            self.image.set_array(self.board)
            return self.image,

        fig = pyplot.figure()
        self.image = pyplot.imshow(
            self.board,
            interpolation="nearest",
            cmap=pyplot.cm.gray
        )

        ani = animation.FuncAnimation(
            fig,
            update_animation,
            interval=interval,
            cache_frame_data=False # or save_count=MAX_FRAMES
        )
        pyplot.show()

    def save_last_frame(self, filename):
        if self.use_torch:
            self.board = self.board.cpu().detach().numpy()
        np.save(filename, self.board)

    def show_current_frame(self):
        if self.use_torch:
            self.board = self.board.cpu().detach().numpy()
        self.image = pyplot.imshow(
            self.board,
            interpolation="nearest",
            cmap=pyplot.cm.gray
        )

        pyplot.show()
        # pyplot.savefig('lastframe.png')


class Conway(Automata):
    def __init__(self, shape, board):
        # which neighbors are on (marked with 1s)
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        # GoL Rule:
        rule = [
            [2, 3], # 'on->on': (2,3): "on" neighbours (can't contain 0)
            [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
        ]
        # init automata
        Automata.__init__(self, board, neighborhood, rule)


class Life34(Automata):
    def __init__(self, shape, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[3, 4], [3, 4]]
        Automata.__init__(self, board, neighborhood, rule)


class Amoeba(Automata):
    def __init__(self, shape, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[1, 3, 5, 8], [3, 5, 7]]
        Automata.__init__(self, shape, board, neighborhood, rule)


class Anneal(Automata):
    def __init__(self, shape, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[3, 5, 6, 7, 8], [4, 6, 7, 8]]
        Automata.__init__(self, board, neighborhood, rule)


class Bugs(Automata):
    def __init__(self, shape, board):
        neighborhood = np.ones((11, 11))
        rule = [
            np.arange(34, 59), # 'on->on': (2,3): "on" neighbours (can't contain 0)
            np.arange(34, 46)  # 'off->on': (3,): "on" neighbours (can't contain 0)
        ]
        Automata.__init__(self, board, neighborhood, rule)


class Globe(Automata):
    def __init__(self, shape, board):
        neighborhood = np.ones((10, 1)) # TODO: something wrong here
        rule = [
            np.arange(34, 59),
            np.arange(34, 46)
        ]
        Automata.__init__(self, board, neighborhood, rule)


def main_other_automata(
        automata_class,
        shape = (256, 256),
        density=0.5,
        seed = 16
    ):

    rng = np.random.default_rng(seed)
    board = rng.uniform(0, 1, shape)
    board = board < density

    automata = automata_class(board)

    # interval=200 # ms
    interval = 0 # as fast as possible
    automata.animate(interval)

    # automata.benchmark(iterations=100)

def main_gol(
        shape_x = 16,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
        iterations=100,
        torus = True,
        animate = False, # if False do benchmark
        show_last_frame = False, # only applicable for benchmark
        save_last_frame = None, # only applicable for benchmark
        use_fft = False,
        torch_device = None,
    ):

    shape = (shape_x, shape_x)

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
            shape_x//2-sq//2:shape_x//2+sq//2,
            shape_x//2-sq//2:shape_x//2+sq//2
        ] = 1 # alive
    else:
        assert initial_state.endswith('.npy')
        board = np.load(initial_state)

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

    # init automata
    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = torus,
        use_fft = use_fft,
        torch_device = torch_device,
    )


    if animate:
        # Animate automata
        # interval = 200 # ms
        interval = 0 # as fast as possible
        automata.animate(interval) #ms
    else:
        # Benchmark automata
        automata.benchmark(iterations)
        if show_last_frame:
            automata.show_current_frame()
        if save_last_frame:
            automata.save_last_frame(save_last_frame)


if __name__ == "__main__":

    ##############
    # CONWAY GAME OF LIFE
    #
    main_gol(
        shape_x = 2**10, #1024,
        initial_state = 'random', # 'square', 'filenmae.npy'
        density = 0.5, # only used with initial_state=='random'
        seed = 123, # only used with initial_state=='random'
        iterations=1000,
        torus = True,
        animate = False,
        show_last_frame = False, # only applicable for benchmark
        save_last_frame = False, # '100k.npy'
        use_fft = False, # conv2d (more efficient)
        # torch_device = 'cpu', # torch cpu
        torch_device = 'cuda', # torch cuda
        # torch_device = 'mps', # torch mps
        # torch_device = None, # numpy
    )
    #
    ##############

    ##############
    # BENCHMARKS
    #
    # Benchmark 1000 iters, torus=True, conv2d
    # Numpy (M1):                   31 Hz
    # Torch mps (M1):              259 Hz
    # Torch cuda (RTX 3090 Ti):   3294 Hz
    #
    # Benchmark 1000 iters, torus=True, fft (less efficient)
    # Numpy (M1):                   24 Hz
    # Torch mps (M1):              186 Hz
    # Torch cuda (RTX 3090 Ti):   1196 Hz
    ##############

    ##############
    # TESTS
    #
    # test_bugs()
    # test_torch()
    #
    ##############

    ##############
    # OTHER AUTOMATA
    #
    # main_other_automata(
    #     # automata_class = Bugs,
    #     # automata_class = Conway,
    #     # automata_class = Life34,
    #     # automata_class = Amoeba,
    #     # automata_class = Anneal,
    #     automata_class = Globe, # TODO: something wrong here
    #     shape = (256, 256),
    #     density=0.5,
    #     seed = 16,
    # )
    #
    ##############