'''
this code is from:
https://gist.github.com/njbbaer/4da02e2960636d349e9bae7ae43c213c
but see also:
- julia: https://rivesunder.github.io/SortaSota/2021/09/27/faster_life_julia.html
- carle: https://github.com/rivesunder/carle
'''

import time
import numpy as np
from numpy.fft import fft2, ifft2
from matplotlib import pyplot, animation



class Automata:
    '''
    shape: must be 2d and power of 2 to make things efficient
    board: initial configuration (binary)
    neighborhood: who are my neighbors (maked with 1s)
    torus: rolling over the boundaries (https://en.wikipedia.org/wiki/Torus)
    '''
    def __init__(self, shape, board, neighborhood, rule, torus=True):
        self.board = board
        self.height, self.width = self.shape = self.board.shape
        self.torus = torus

        nh, nw = neighborhood.shape # say (3,3) for Conway's GoL

        # create the kernal (init as zero)
        self.kernal = np.zeros(shape) # kernal has same shape of board, say (256,256)
        # put neighborhood mask in the middle (everything else is 0.)
        self.kernal[
            (shape[0] - nh - 1) // 2 : (shape[0] + nh) // 2,
            (shape[1] - nw - 1) // 2 : (shape[1] + nw) // 2
        ] = neighborhood

        self.kernal_ft = fft2(self.kernal) # same shape but floating numbers

        self.rule = rule

    '''
    Main count_real operation in a single time-step
    '''
    def fft_convolve2d(self):
        board_ft = fft2(self.board) # same shape but floating numbers

        # inverted fft2 (complex numbers)
        inverted = ifft2(board_ft * self.kernal_ft)
        # get the real part of the complex argument (real number)
        count_real = np.real(inverted)

        # rolling over the boundaries (https://en.wikipedia.org/wiki/Torus)
        if self.torus:
            count_real = np.roll(count_real, - int(self.height / 2) + 1, axis=0)
            count_real = np.roll(count_real, - int(self.width / 2) + 1, axis=1)

        counts_int = np.rint(count_real)

        # double check
        # counts_round = count_real.round()
        # assert np.array_equal(counts_int, counts_round)
        # return counts_round

        return counts_int


    def update_board(self, intervals=1):
        for _ in range(intervals):
            # resulting of count_real - sanme shape - number of alive cells
            count_alive = self.fft_convolve2d()
            shape = count_alive.shape
            new_board = np.zeros(shape)
            new_board[
                np.where(
                    np.isin(count_alive, self.rule[0]).reshape(shape)
                    &
                    (self.board == 1)
                )
            ] = 1
            new_board[
                np.where(
                    np.isin(count_alive, self.rule[1]).reshape(shape)
                    &
                    (self.board == 0)
                )
            ] = 1

            self.board = new_board


    def benchmark(self, interations):
        start = time.process_time()
        self.update_board(interations)
        print("Performed", interations, "iterations of", self.board.shape, "cells in",
              time.process_time() - start, "seconds")


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
        neighborhood = np.ones((10, 1))
        rule = [np.arange(34, 59), np.arange(34, 46)]
        Automata.__init__(self, shape, board, neighborhood, rule)


class Animation:
    def __init__(self, automata, interval=100):
        self.automata = automata
        fig = pyplot.figure()
        self.image = pyplot.imshow(self.automata.board, interpolation="nearest",
                                   cmap=pyplot.cm.gray)
        ani = animation.FuncAnimation(fig, self.animate, interval=interval)
        pyplot.show()


    def animate(self, *args):
        self.automata.update_board()
        self.image.set_array(self.automata.board)
        return self.image,

def test_bugs():
    density=0.5
    seed = 16
    shape = (256, 256)
    rng = np.random.default_rng(seed)
    board = rng.uniform(0, 1, shape)
    board = board < density
    automata = Bugs(shape, board)
    # automata.animate(interval=200) #ms
    automata.benchmark(interations=100)

def test_torch():
    # import timeit
    import torch
    print('cuda available:', torch.cuda.is_available())
    print('mps available:', torch.backends.mps.is_available())
    mps_device = torch.device("mps")
    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # x = torch.rand((10000, 10000), dtype=torch.float32)
    # Any operation happens on the GPU
    y = x * 2
    print(y)

    '''
    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)
    '''


def main():

    random_init = True
    shape_x = 128
    shape = (shape_x, shape_x)


    if random_init:
        seed = 123
        density = 0.5

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

    neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # GoL Rule:
    # automata = Conway(shape=shape,board=board)
    rule = [
        [2, 3], # 'on->on': (2,3): "on" neighbours (can't contain 0)
        [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
    ]

    # exploring other rules:
    # rule = [
    #     [2], # 'on->on': "on" neighbours (can't contain 0)
    #     [1]  # 'off->on':   "on" neighbours (can't contain 0)
    # ]

    # init automata
    automata = Automata(shape, board, neighborhood, rule)

    # Other automata
    # automata = Bugs((256, 256), density=0.5)
    # automata = Life34((256, 256), density=0.12)
    # automata = Amoeba((256, 256), density=0.18)
    # automata = Anneal((256, 256), density=0.5)

    # Animate automata
    automata.animate(interval=200) #ms

    # Benchmark automata
    # automata.benchmark(interations=1000)


if __name__ == "__main__":
    # test_bugs()
    # test_torch()

    main()