from gol.pure.automata import Automata
import numpy as np

class Amoeba(Automata):
    def __init__(self, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[1, 3, 5, 8], [3, 5, 7]]
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

    automata = Amoeba(board)

    automata.animate(interval = 0) # ms
    # automata.benchmark(iterations=100)

if __name__ == "__main__":
    main()