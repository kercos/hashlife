import time
from collections import Counter
import random

def random_matrix(size, seed=None):
    random.seed(seed)
    return Counter([
        (x,y)
        for x in range(size)
        for y in range(size)
        if bool(random.getrandbits(1)) # flip a coin
    ])


gol_rule = {
    'on->on': (2,3), # "on" neighbours (can't contain 0)
    'off->on': (3,) # "on" neighbours (can't contain 0)
}

'''
on_cells: a collection of "on" cells stored as (x,y) pairs,
'''
def life(on_cells, rule):

    on_on = rule['on->on']
    off_on = rule['off->on']

    # count how many "on" cells are around a given cell (x,y)
    # all cells not in count have 0 "on" neighbours
    on_neighbours = Counter(
        [
            (x+a, y+b)
            for x,y in on_cells
            for a in [-1,0,1]
            for b in [-1,0,1]
        ]
    )

    return Counter(
        [
            p for p in on_neighbours
            if (
                # "off" cells with `off_on` alive (on) neightbours turn "on"
                (p not in on_cells and on_neighbours[p] in off_on)
                or
                (p in on_cells and on_neighbours[p] in on_on)
                # "on" cells with `on_on` alive (on) neightbours stay "on"
            )
        ]
    )

if __name__ == "__main__":

    # number of iterations
    iters = 10

    # "on" cells (x,y)
    on_cells = random_matrix(1000, seed=123)

    init_t = time.perf_counter()

    for _ in range(iters):
        on_cells = life(on_cells, gol_rule)

    t = time.perf_counter() - init_t
    print(f"took {t:.1f}s")