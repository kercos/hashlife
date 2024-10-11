from gol.pure.automata import Automata
import numpy as np

class Conway(Automata):
    def __init__(self, board):
        # which neighbors are on (marked with 1s)
        neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        # GoL Rule:
        rule = [
            [2, 3], # 'on->on': (2,3): "on" neighbours (can't contain 0)
            [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
        ]
        # init automata
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

    automata = Conway(board)

    automata.animate(interval = 0) # ms
    # automata.benchmark(iterations=100)

if __name__ == "__main__":
    main()