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
    def __init__(self, shape, board, neighborhood, rule, torus=True, torch_device = None):

        assert (
            0 not in rule[0]
            and
            0 not in rule[1]
        ), "Rule cannot contain zeros"

        assert (
            torch_device in [None, "cpu", "cuda", "mps"]
        ), f"torch_device ({torch_device}) not recognized"

        self.torch_device = torch_device
        self.use_torch = torch_device is not None

        self.board = board

        self.shape_x, _, = self.height, self.width = self.shape = self.board.shape

        assert self.height == self.width

        self.minus_shape_x_half_plus_one =  - int(self.shape_x / 2) + 1

        self.torus = torus

        nh, nw = neighborhood.shape # say (3,3) for Conway's GoL

        # create the kernal (init as zero)
        # put neighborhood mask in the middle (everything else is 0.)
        # TODO check:
        # - not always perfectly centered
        # - e.g., with 3x3 neighborhood and grid size being even (power of two)
        self.kernal = np.zeros(shape) # kernal has same shape of board, say (256,256)
        self.kernal[
            (shape[0] - nh - 1) // 2 : (shape[0] + nh) // 2,
            (shape[1] - nw - 1) // 2 : (shape[1] + nw) // 2
        ] = neighborhood

        self.rule = rule

        # convert arrays to tensor if using torch
        if self.use_torch:
            # print('cuda available:', torch.cuda.is_available())
            # print('mps available:', torch.backends.mps.is_available())
            torch_device = torch.device(torch_device)
            torch.set_default_device(torch_device)
            self.rule = [torch.IntTensor(r).to(torch_device) for r in self.rule] # need to convert each rule to int tensor
            self.board = torch.from_numpy(self.board).to(torch_device)
            self.kernal = torch.from_numpy(self.kernal).float().to(torch_device)
            self.kernal_ft = torch_fft2(self.kernal)
        else:
            self.kernal_ft = np_fft2(self.kernal) # same shape but floating numbers




    '''
    Main count_real operation in a single time-step
    '''
    def fft_convolve2d(self):
        # fft2 same shape but floating numbers
        board_ft = np_fft2(self.board)

        # inverted fft2 (complex numbers)
        inverted = np_ifft2(board_ft * self.kernal_ft)

        # get the real part of the complex argument (real number)
        count_real = np.real(inverted)

        # round real part to closest integer
        counts_int = np.rint(count_real)

        # double check
        # counts_round = count_real.round()
        # assert np.array_equal(counts_int, counts_round)
        # return counts_round

        # rolling over the boundaries
        # see https://en.wikipedia.org/wiki/Torus
        if self.torus:
            counts_int = np.roll(
                counts_int,
                self.minus_shape_x_half_plus_one,
                axis=(0,1)
            )

        return counts_int

    '''
    Main count_real operation in a single time-step (torch version)
    '''
    def fft_convolve2d_torch(self):
        # fft2 same shape but floating numbers
        board_ft = torch_fft2(self.board)

        # inverted fft2 (complex numbers)
        inverted = torch_ifft2(board_ft * self.kernal_ft)

        # get the real part of the complex argument (real number)
        count_real = torch.real(inverted)

        # round real part to closest integer
        counts_int = torch.round(count_real).int()

        # double check
        # counts_round = count_real.round()
        # assert np.array_equal(counts_int, counts_round)
        # return counts_round

        # rolling over the boundaries
        # see https://en.wikipedia.org/wiki/Torus
        if self.torus:
            counts_int = torch.roll(
                counts_int,
                (
                    self.minus_shape_x_half_plus_one,
                    self.minus_shape_x_half_plus_one
                ), # need tuple here because torch behaves differently from numpy apparently
                dims=(0,1)
            )

        return counts_int

    def update_board(self):
        # counting number of alive cells in neighbourhood (same shape)
        count_ones_neighbours = self.fft_convolve2d()
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

    def update_board_torch(self):
        # counting number of alive cells in neighbourhood (same shape)
        count_ones_neighbours = self.fft_convolve2d_torch()
        board_ones = self.board == 1
        board_zeros = ~ board_ones # negation

        new_board = torch.zeros(self.shape)
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

    def benchmark(self, iterations):
        start = time.process_time()

        if self.use_torch:
            for _ in range(iterations):
                self.update_board_torch()

        else:
            for _ in range(iterations):
                self.update_board()

        ellapsed = time.process_time() - start
        print(
            "Performed", iterations,
            "iterations of", self.board.shape,
            "cells in", ellapsed, "seconds"
        )


    def animate(self, interval=100):

        def update_animation(*args):
            self.update_board()
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
        Automata.__init__(self, shape, board, neighborhood, rule)


class Life34(Automata):
    def __init__(self, shape, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[3, 4], [3, 4]]
        Automata.__init__(self, shape, board, neighborhood, rule)


class Amoeba(Automata):
    def __init__(self, shape, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[1, 3, 5, 8], [3, 5, 7]]
        Automata.__init__(self, shape, board, neighborhood, rule)


class Anneal(Automata):
    def __init__(self, shape, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[3, 5, 6, 7, 8], [4, 6, 7, 8]]
        Automata.__init__(self, shape, board, neighborhood, rule)


class Bugs(Automata):
    def __init__(self, shape, board):
        neighborhood = np.ones((11, 11))
        rule = [
            np.arange(34, 59), # 'on->on': (2,3): "on" neighbours (can't contain 0)
            np.arange(34, 46)  # 'off->on': (3,): "on" neighbours (can't contain 0)
        ]
        Automata.__init__(self, shape, board, neighborhood, rule)


class Globe(Automata):
    def __init__(self, shape, board):
        neighborhood = np.ones((10, 1)) # TODO: something wrong here
        rule = [
            np.arange(34, 59),
            np.arange(34, 46)
        ]
        Automata.__init__(self, shape, board, neighborhood, rule)


class Animation:
    def __init__(self, automata, interval=100):
        self.automata = automata
        fig = pyplot.figure()
        self.image = pyplot.imshow(
            self.automata.board,
            interpolation="nearest",
            cmap=pyplot.cm.gray
        )
        animation.FuncAnimation(
            fig,
            self.animate,
            interval=interval
        )
        pyplot.show()


    def animate(self, *args):
        self.automata.update_board()
        self.image.set_array(self.automata.board)
        return self.image,


def test_torch():
    # import timeit
    import torch
    print('cuda available:', torch.cuda.is_available())
    print('mps available:', torch.backends.mps.is_available())
    torch_device = torch.device("mps")
    torch.set_default_device(torch_device)

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=torch_device)

    # x = torch.rand((10000, 10000), dtype=torch.float32)
    # Any operation happens on the GPU

    board = torch.zeros((128,128))
    board = torch.fft.fft2(board)
    # board = torch.fft.ifft2
    board = (board > 2) & (board<5)

    # board.cpu().numpy() // check this, you may need to copy it to cpu

    '''
    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)
    '''

def test_torch_ca():
    # from https://github.com/moritztng/cellular/
    import torch
    import torch.nn.functional as torch_functions

    class Universe:
        def __init__(self, name, state, rule, colors):
            self.name = name
            self.state = state
            self.rule = rule
            self.colors = colors

        def step(self):
            self.state = self.rule(self.state)
    class GameOfLife:
        def __init__(self, device):
            self.parameters = torch.zeros((2, 2, 3, 3), dtype=torch.float32, device=device)
            self.parameters[1, 1, :, :] = 1
            self.parameters[1, 1, 1, 1] = 9

        def __call__(self, state):
            next_state = torch_functions.pad(state, (1, 1, 1, 1), mode="circular")
            next_state = torch_functions.conv2d(next_state, self.parameters)
            next_state = ((next_state == 3) + (next_state == 11) + (next_state == 12)).to(torch.float32)
            next_state[:, 0, :, :] = 1 - next_state[:, 1, :, :]
            return next_state



def main_other_automata(
        automata_class,
        shape = (256, 256),
        density=0.5,
        seed = 16
    ):

    rng = np.random.default_rng(seed)
    board = rng.uniform(0, 1, shape)
    board = board < density

    automata = automata_class(shape, board)

    # interval=200 # ms
    interval = 0 # as fast as possible
    automata.animate(interval)

    # automata.benchmark(iterations=100)

def main_gol(
        random_init = True, shape_x = 16,
        animate = False, # if False do benchmark
        seed = 123, density = 0.5, # only used on random_init
        torus = True,
        torch_device = None
    ):

    shape = (shape_x, shape_x)

    if random_init:
        # initialize random generator
        rng = np.random.default_rng(seed)
        board = rng.uniform(0, 1, shape)
        board = board < density
    else:
        sq = 2 # alive square size in the middle of the board
        assert sq % 2 == 0
        board = np.zeros(shape)
        board[
            shape_x//2-sq//2:shape_x//2+sq//2,
            shape_x//2-sq//2:shape_x//2+sq//2
        ] = 1 # alive

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
        shape = shape,
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = torus,
        torch_device = torch_device
    )


    if animate:
        # Animate automata
        # interval = 200 # ms
        interval = 0 # as fast as possible
        automata.animate(interval) #ms
    else:
        # Benchmark automata
        automata.benchmark(iterations=100)


if __name__ == "__main__":
    # test_bugs()
    # test_torch()
    # test_torch_ca()

    main_gol(
        random_init = True,
        shape_x = 1024,
        seed = 123, # only used on random_init
        density = 0.5, # only used on random_init
        animate = False,
        torus = True,
        torch_device = 'mps'
    )
    # Numpy:        Performed 100 iterations of (1024, 1024) cells in 4.27 seconds
    # Torch mps:    Performed 100 iterations of (1024, 1024) cells in 0.94 seconds

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