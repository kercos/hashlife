'''
https://gist.github.com/njbbaer/4da02e2960636d349e9bae7ae43c213c
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
    '''
    def __init__(self, shape, density, neighborhood, rule, seed=None):
        self.seed = seed
        # initialize random generator
        rng = np.random.default_rng(self.seed)

        self.board = rng.uniform(0, 1, shape)
        self.board = self.board < density

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
    def __init__(self, shape, density, seed=None):
        # which neighbors are on (marked with 1s)
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        # GoL Rule:
        rule = [
            [2, 3], # 'on->on': (2,3): "on" neighbours (can't contain 0)
            [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
        ]
        # init automata
        Automata.__init__(self, shape, density, neighborhood, rule, seed)


class Life34(Automata):
    def __init__(self, shape, density, seed=None):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[3, 4], [3, 4]]
        Automata.__init__(self, shape, density, neighborhood, rule, seed)


class Amoeba(Automata):
    def __init__(self, shape, density, seed=None):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[1, 3, 5, 8], [3, 5, 7]]
        Automata.__init__(self, shape, density, neighborhood, rule, seed)


class Anneal(Automata):
    def __init__(self, shape, density, seed=None):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[3, 5, 6, 7, 8], [4, 6, 7, 8]]
        Automata.__init__(self, shape, density, neighborhood, rule, seed)


class Bugs(Automata):
    def __init__(self, shape, density, seed=None):
        neighborhood = np.ones((11, 11))
        rule = [np.arange(34, 59), np.arange(34, 46)]
        Automata.__init__(self, shape, density, neighborhood, rule, seed)


class Globe(Automata):
    def __init__(self, shape, density, seed=None):
        neighborhood = np.ones((10, 1))
        rule = [np.arange(34, 59), np.arange(34, 46)]
        Automata.__init__(self, shape, density, neighborhood, rule, seed)


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
    # Create automata
    # automata = Bugs((256, 256), density=0.5)
    automata = Conway((256, 256), density=0.5)
    # automata = Life34((256, 256), density=0.12)
    # automata = Amoeba((256, 256), density=0.18)
    # automata = Anneal((256, 256), density=0.5)

    # Benchmark automata
    # automata.benchmark(interations=100)

    # Animate automata
    automata.animate(interval=100)

def test():
    automata = Conway((32, 32), density=0.5, seed=123)
    automata.animate(interval=100)
    # num_steps = 100
    # init_t = time.perf_counter()
    # for _ in range(num_steps):
    #     automata.update_board()
    # t = time.perf_counter() - init_t
    # print(f"Computation took {t*1000.0:.1f}ms")


if __name__ == "__main__":
    # main()
    test()