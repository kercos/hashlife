from collections import namedtuple
from functools import lru_cache
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# The base quadtree node
# `k` is the level of the node
# `a, b, c, d` are the children of this node (or None, if `k=0`).
# `n` is the number of on cells in this node (useful for bookkeeping and display)
# `hash` is a precomputed hash of this node
# (if we don't do this, Python will recursively compute the hash every time it is needed!)
class Node:
    __slots__ = ["k", "a", "b", "c", "d", "n", "hash"]

    def __init__(self, k, n, hash, a=None, b=None, c=None, d=None):
        self.k = k
        self.n = n
        self.a, self.b, self.c, self.d = a,b,c,d
        self.hash = hash

    def __hash__(self):
        return self.hash

    def equals(self, other):
        if isinstance(other, Node):
            return (
                self.k == other.k and
                self.a == other.a and
                self.b == other.b and
                self.c == other.c and
                self.d == other.d
            )
        return False

    def __repr__(self):
        return f"Node k={self.k}, {1<<self.k} x {1<<self.k}, population {self.n}"

# base level binary nodes
on = Node(k=0, n=1, hash=1)
off = Node(k=0, n=0, hash=0)

mask = (1 << 63) - 1

#####################
# CONSTRUCTORS
#
@lru_cache(maxsize=1024)
def get_zero(k):
    """Return an empty node at level `k`."""
    if k > 0:
        return join(get_zero(k - 1), get_zero(k - 1), get_zero(k - 1), get_zero(k - 1))
    else:
        return off

def construct(pts):
    """
    Turn a list of (x,y) coordinates into a quadtree
    and return the top-level Node.
    """
    # Force start at (0,0)
    min_x = min(*[x for x, y in pts])
    min_y = min(*[y for x, y in pts])
    pattern = {(x - min_x, y - min_y): on for x, y in pts}
    k = 0
    while len(pattern) != 1:
        # bottom-up construction
        next_level = {}
        z = get_zero(k)
        while len(pattern) > 0:
            x, y = next(iter(pattern))
            x, y = x - (x & 1), y - (y & 1)
            # read all 2x2 neighbours, removing from those to work through
            # at least one of these must exist by definition
            a = pattern.pop((x, y), z)
            b = pattern.pop((x + 1, y), z)
            c = pattern.pop((x, y + 1), z)
            d = pattern.pop((x + 1, y + 1), z)
            next_level[x >> 1, y >> 1] = join(a, b, c, d)
        # merge at the next level
        pattern = next_level
        k += 1
    return pad(pattern.popitem()[1])
#
# CONSTRUCTORS
#####################


#####################
# DECONSTRUCTORS
#
def expand(node, x=0, y=0, clip=None, level=0):
    """Turn a quadtree a list of (x,y,gray) triples
    in the rectangle (x,y) -> (clip[0], clip[1]) (if clip is not-None).
    If `level` is given, quadtree elements at the given level are given
    as a grayscale level 0.0->1.0,  "zooming out" the display.
    """

    if node.n == 0:  # quick zero check
        return []
    size = 2 ** node.k
    # bounds check
    if clip is not None:
        if x + size < clip[0] or x > clip[1] or y + size < clip[2] or y > clip[3]:
            return []
    if node.k == level:
        # base case: return the gray level of this node
        gray = node.n / (size ** 2)
        return [(x >> level, y >> level, gray)]
    else:
        # return all points contained inside this node
        offset = size >> 1
        return (
            expand(node.a, x, y, clip, level)
            + expand(node.b, x + offset, y, clip, level)
            + expand(node.c, x, y + offset, clip, level)
            + expand(node.d, x + offset, y + offset, clip, level)
        )
#
# DECONSTRUCTORS
#####################


#####################
# STATIC OPERATIONS
#
def centre(m):
    """
    Return a node at level `k+1`, which is centered on the given quadtree node.
    This basically adds a single-cell frame with off cells (perimeter) to the pattern
    """
    z = get_zero(m.a.k)  # get the right-sized zero node
    return join(
        join(z, z, z, m.a), join(z, z, m.b, z), join(z, m.c, z, z), join(m.d, z, z, z)
    )

def inner(node):
    """
    Return the central portion of a node -- the inverse operation
    of centre()
    """
    return join(node.a.d, node.b.c, node.c.b, node.d.a)

def is_padded(node):
    """
    True if the pattern is surrounded by at least one sub-sub-block of
    empty space.
    """
    return (
        node.a.n == node.a.d.d.n
        and node.b.n == node.b.c.c.n
        and node.c.n == node.c.b.b.n
        and node.d.n == node.d.a.a.n
    )

def crop(node):
    """
    Repeatedly take the inner node, until all padding is removed.
    """
    if node.k <= 3 or not is_padded(node):
        return node
    else:
        return crop(inner(node))

def pad(node):
    """
    Repeatedly centre a node, until it is fully padded.
    """
    if node.k <= 3 or not is_padded(node):
        return pad(centre(node))
    else:
        return node

def life(a, b, c, d, E, f, g, h, i):
    """The standard life rule, taking eight neighbours and a centre cell E.
    Returns on if should be on, and off otherwise."""
    outer = sum([t.n for t in [a, b, c, d, f, g, h, i]])
    return on if (E.n and outer == 2) or outer == 3 else off # GoL rule

def life_4x4(m):
    """
    Return the next generation of a $k=2$ (i.e. 4x4) cell.
    To terminate the recursion, at the base level,
    if we have a $k=2$ 4x4 block,
    we can compute the 2x2 central successor by iterating over all
    the 3x3 sub-neighbourhoods of 1x1 cells using the standard Life rule.
    """
    na = life(m.a.a, m.a.b, m.b.a, m.a.c, m.a.d, m.b.c, m.c.a, m.c.b, m.d.a)  # AD
    nb = life(m.a.b, m.b.a, m.b.b, m.a.d, m.b.c, m.b.d, m.c.b, m.d.a, m.d.b)  # BC
    nc = life(m.a.c, m.a.d, m.b.c, m.c.a, m.c.b, m.d.a, m.c.c, m.c.d, m.d.c)  # CB
    nd = life(m.a.d, m.b.c, m.b.d, m.c.b, m.d.a, m.d.b, m.c.d, m.d.c, m.d.d)  # DA
    return join(na, nb, nc, nd)

@lru_cache(maxsize=2 ** 24)
def join(a, b, c, d):
    """
    Combine four children at level `k-1` to a new node at level `k`.
    If this is cached, return the cached node.
    Otherwise, create a new node, and add it to the cache.
    """
    n = a.n + b.n + c.n + d.n
    nhash = (
        a.k
        + 2
        + 5131830419411 * a.hash
        + 3758991985019 * b.hash
        + 8973110871315 * c.hash
        + 4318490180473 * d.hash
    ) & mask
    return Node(a.k+1, n, nhash, a, b, c, d)

@lru_cache(maxsize=2 ** 24)
def successor(m, j=None):
    """
    Return the 2**k-1 x 2**k-1 successor, 2**j generations in the future,
    where j <= k - 2, caching the result.
    Therefore for a 8x8 (k=3) successor (k=2), up to 1 generation in the future
        (TODO: to confirm)
    """
    if m.n == 0:  # empty
        return m.a
    elif m.k == 2:  # base case
        s = life_4x4(m)
    else:
        j = m.k - 2 if j is None else min(j, m.k - 2)
        c1 = successor(join(m.a.a, m.a.b, m.a.c, m.a.d), j)
        c2 = successor(join(m.a.b, m.b.a, m.a.d, m.b.c), j)
        c3 = successor(join(m.b.a, m.b.b, m.b.c, m.b.d), j)
        c4 = successor(join(m.a.c, m.a.d, m.c.a, m.c.b), j)
        c5 = successor(join(m.a.d, m.b.c, m.c.b, m.d.a), j)
        c6 = successor(join(m.b.c, m.b.d, m.d.a, m.d.b), j)
        c7 = successor(join(m.c.a, m.c.b, m.c.c, m.c.d), j)
        c8 = successor(join(m.c.b, m.d.a, m.c.d, m.d.c), j)
        c9 = successor(join(m.d.a, m.d.b, m.d.c, m.d.d), j)

        if j < m.k - 2:
            s = join(
                (join(c1.d, c2.c, c4.b, c5.a)),
                (join(c2.d, c3.c, c5.b, c6.a)),
                (join(c4.d, c5.c, c7.b, c8.a)),
                (join(c5.d, c6.c, c8.b, c9.a)),
            )
        else:
            s = join(
                successor(join(c1, c2, c4, c5), j),
                successor(join(c2, c3, c5, c6), j),
                successor(join(c4, c5, c7, c8), j),
                successor(join(c5, c6, c8, c9), j),
            )
    return s
#
# STATIC OPERATIONS
#####################


#####################
# TIME DYNAMICS
#
def advance(node, n):
    """Advance node by exactly n generations, using
    the binary expansion of n to find the correct successors"""
    if n == 0:
        return node
    bits = []
    # get the binary expansion, and pad sufficiently
    while n > 0:
        bits.append(n & 1)
        n = n >> 1
        node = centre(node)

    # apply the successor rule
    for k, bit in enumerate(reversed(bits)):
        j = len(bits) - k - 1
        if bit:
            node = successor(node, j)
    return crop(node)

def ffwd(node, n):
    """Advance as quickly as possible, taking n
    giant leaps"""
    gens = 0
    for i in range(n):
        node = pad(node)
        gens += 1 << (node.k - 2)
        node = successor(node)
    return node, gens

def get_gen_for_giant_leaps(k, n):
    """Get the number of generation equivalent
    for n giant leaps for a given node of given k
    See `ffwd` function above (only gens are returned here)
    TODO: double check
    """
    gens = 0
    for _ in range(n):
        gens += 1 << (k - 2)
        k = k + 1 # tricky as this may depend on padding
    return gens
#
# TIME DYNAMICS
#####################


#####################
# RENDERING
#
def render_img(
        node,
        level=0,
        crop=True, # by default render the smallest box containing alive cells
        offset = None, # only when crop is False (moving things to center)
        name=None, filepath=None,
        show=True, force_show=False):
    """
    Utility to show a point collection as an image in Matplotlib
    """
    pts = expand(node, level=level) # triplets with points and gray values (only > 0)
    pts = np.array(pts)
    pts[:, 0] -= np.min(pts[:, 0])
    pts[:, 1] -= np.min(pts[:, 1])

    if crop:
        max_x = int(np.max(pts[:, 0] + 1))
        max_y = int(np.max(pts[:, 1] + 1))
        grays = np.zeros((max_y, max_x))
    else:
        size = 2 ** node.k
        grays = np.zeros((size,size))

    if offset is not None:
        assert crop==False, "offset only valid when crop is False"
        offset_y, offset_x = offset
        for x, y, g in pts:
            grays[int(y+offset_y), int(x+offset_x)] = g
    else:
        for x, y, g in pts:
            grays[int(y), int(x)] = g

    if filepath:
        fig = plt.figure()
        # plt.imshow(grays, cmap="bone")
        plt.imshow(
            grays,
            interpolation="nearest",
            cmap=plt.cm.gray
        )
        plt.axis("off")
        plt.savefig(
            filepath,
            bbox_inches='tight', pad_inches=0 # no frame
        )
        plt.close(fig) # so it's not shown

    if show:
        plt.figure(name, figsize=(5, 5))
        plt.axis("off")
        # plt.imshow(grays, cmap="bone")
        plt.imshow(
            grays,
            interpolation="nearest",
            cmap=plt.cm.gray
        )
        if force_show:
            plt.show()
    return grays

def print_node(node):
    """
    Print out a node, fully expanded
    """
    points = expand(crop(node))
    px, py = 0, 0
    for x, y, gray in sorted(points, key=lambda x: (x[1], x[0])):
        while y > py:
            print()
            py += 1
            px = 0
        while x > px:
            print(" ", end="")
            px += 1
        print("*", end="")
#
# RENDERING
#####################