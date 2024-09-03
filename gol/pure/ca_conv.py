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


def fft_convolve2d(board, kernal):
    board_ft = fft2(board)
    kernal_ft = fft2(kernal)
    height, width = board_ft.shape
    convolution = np.real(ifft2(board_ft * kernal_ft))
    convolution = np.roll(convolution, - int(height / 2) + 1, axis=0)
    convolution = np.roll(convolution, - int(width / 2) + 1, axis=1)
    return convolution.round()


class Automata:
    '''
    shape: must be 2d and power of 2 to make things efficient
    neighborhood: who are my neighbors (maked with 1s)
    configuration: initial configuration
    '''
    def __init__(self, shape, board, neighborhood, rule):
        self.board = board
        n_height, n_width = neighborhood.shape # say (3,3) for Conway's GoL
        self.kernal = np.zeros(shape) # kernal has same shape, say (256,256)
        self.kernal[
            (shape[0] - n_height - 1) // 2 : (shape[0] + n_height) // 2,
            (shape[1] - n_width - 1) // 2 : (shape[1] + n_width) // 2
        ] = neighborhood

        self.rule = rule


    def update_board(self, intervals=1):
        for i in range(intervals):
            convolution = fft_convolve2d(self.board, self.kernal)
            shape = convolution.shape
            new_board = np.zeros(shape)
            new_board[np.where(np.in1d(convolution, self.rule[0]).reshape(shape)
                               & (self.board == 1))] = 1
            new_board[np.where(np.in1d(convolution, self.rule[1]).reshape(shape)
                               & (self.board == 0))] = 1
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
        self.image = pyplot.imshow(self.board, interpolation="nearest",
                                   cmap=pyplot.cm.gray)
        ani = animation.FuncAnimation(fig, update_animation, interval=interval)
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
        rule = [np.arange(34, 59), np.arange(34, 46)]
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


def main():

    random_init = False
    shape_x = 256
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
    # rule = [
    #     [2, 3], # 'on->on': (2,3): "on" neighbours (can't contain 0)
    #     [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
    # ]

    # exploring other rules:
    rule = [
        [2], # 'on->on': "on" neighbours (can't contain 0)
        [1]  # 'off->on':   "on" neighbours (can't contain 0)
    ]

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
    # automata.benchmark(interations=100)


if __name__ == "__main__":
    main()