from gol.pure.automata import Automata
import numpy as np

class Globe(Automata):
    def __init__(self, board):
        neighborhood = np.ones((10, 1)) # TODO: something wrong here
        rule = [
            np.arange(34, 59),
            np.arange(34, 46)
        ]
        Automata.__init__(self, board, neighborhood, rule)

def main(
        shape_x = 256,
        density=0.5,
        seed = 123
    ):

    rng = np.random.default_rng(seed)
    shape = (shape_x, shape_x)
    board = rng.uniform(0, 1, shape)
    board = board < density

    automata = Globe(board) # TODO: something wrong here

    automata.animate(interval = 0) # ms
    # automata.benchmark(iterations=100)

if __name__ == "__main__":
    main()