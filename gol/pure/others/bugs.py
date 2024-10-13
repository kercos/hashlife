from gol.pure.automata import Automata
import numpy as np

class Bugs(Automata):
    def __init__(self, board):
        neighborhood = np.ones((11, 11))
        rule = [
            np.arange(34, 59), # 'on->on': (2,3): "on" neighbours (can't contain 0)
            np.arange(34, 46)  # 'off->on': (3,): "on" neighbours (can't contain 0)
        ]
        Automata.__init__(self, board, neighborhood, rule)

def main(
        size = 256,
        density=0.5,
        seed = 123
    ):

    rng = np.random.default_rng(seed)
    shape = (size, size)
    board = rng.uniform(0, 1, shape)
    board = board < density

    automata = Bugs(board)

    automata.animate(interval = 0) # ms
    # automata.benchmark(iterations=100)

if __name__ == "__main__":
    main()