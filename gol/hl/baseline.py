from collections import Counter

def baseline_life(pts):
    """
    The baseline implementation of the Game of Life. Takes a list/set
    of (x,y) on cells, and returns a new set of on cells in the next
    generation.
    """
    ns = Counter(
        [
            (x + a, y + b)
            for x, y in pts
            for a in [-1, 0, 1]
            for b in [-1, 0, 1]
        ]
    )
    return Counter(
        [
            p
            for p in ns
            if ns[p] == 3 or (ns[p] == 4 and p in pts)
        ]
    )

