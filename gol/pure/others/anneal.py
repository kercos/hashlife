from gol.pure.automata import Automata
import numpy as np

class Anneal(Automata):
    def __init__(self, board):
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        rule = [[3, 5, 6, 7, 8], [4, 6, 7, 8]]
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

    automata = Anneal(board)

    automata.animate(interval = 0) # ms
    # automata.benchmark(iterations=100)

if __name__ == "__main__":
    main()