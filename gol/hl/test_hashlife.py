from gol.hl.hashlife import (
    join, successor,
    on, off,
    construct, centre, expand, inner,
    pad, crop, is_padded, get_zero,
    advance,
    ffwd,
)
from gol.hl.baseline import baseline_life
from gol.hl.lifeparsers import autoguess_life_file
from itertools import product
import os
from functools import lru_cache


def test_bootstrap():
    # try to generate all 4x4 successors
    def product_tree(pieces):
        return [join(a, b, c, d) for a, b, c, d in product(pieces, repeat=4)]

    # check cache is working
    for i in range(2):
        boot_2x2 = product_tree([on, off])
        boot_4x4 = product_tree(boot_2x2)
        centres = {p: successor(p, 1) for p in boot_4x4}

        assert join.cache_info().currsize == 65536 + 16
        assert successor.cache_info().currsize == 65536


test_fname = "input/lifep/gun30.lif"
test_pattern, _ = autoguess_life_file(test_fname)


def test_advance():
    pat_node = construct(test_pattern)
    for i in range(32):
        node = advance(pat_node, i)
        validate_tree(node)

    for i in range(4):
        node, gens = ffwd(pat_node, i)
        validate_tree(node)

    assert advance(get_zero(8), 8).n == 0
    assert ffwd(get_zero(8), 4)[0].n == 0


def test_ffwd_large():
    pat, _ = autoguess_life_file("input/lifep/breeder.lif")
    ffwd(construct(pat), 64)


def test_get_zero():
    for i in range(32):
        z = get_zero(i)
        assert z.k == i
        assert z.n == 0


def align(pts):
    min_x = min(*[x for x, y in pts])
    min_y = min(*[y for x, y in pts])
    return sorted([(x - min_x, y - min_y) for x, y in pts])


def same_pattern(pt_a, expanded):
    return align(pt_a) == align([(x, y) for x, y, gray in expanded])


def test_gray():
    pat, _ = autoguess_life_file("input/hl_lifep/breeder.lif")
    node = construct(pat)
    total_on = len(expand(node))
    for l in range(6):
        expansion = expand(node, level=l)
        gray_sum = sum([g for (x, y, g) in expansion])
        assert gray_sum == total_on / (2 ** (l * 2))


def verify_clipped(node, x1, y1, x2, y2):
    pts = expand(node, clip=(x1, y1, x2, y2))
    assert all([x >= x1 and x <= x2 and y > y1 and y < y2 for x, y in pts])


def test_clip():
    pat, _ = autoguess_life_file("input/hl_lifep/breeder.lif")
    node = construct(pat)
    verify_clipped(node, 0, 0, 1600, 1600)
    verify_clipped(node, 0, 0, 160, 160)
    verify_clipped(node, 40, 40, 1600, 1600)
    verify_clipped(node, 40, 40, 160, 160)


def verify_baseline(pat, n):
    from gol.hl.hashlife import construct, expand, advance
    from gol.hl.test_hashlife import same_pattern
    node = construct(pat)
    if not same_pattern(pat, expand(node)):
        return False
    for i in range(n):
        advanced = advance(node, i)
        if not same_pattern(pat, expand(advanced)):
            return False
        pat = baseline_life(pat)
    return True


def test_all_patterns():
    lifep_dir = "input/lifep/"
    from tqdm import tqdm
    for pat_fname in tqdm(sorted(os.listdir(lifep_dir))):
    # for pat_fname in sorted(os.listdir(lifep_dir)):
        if pat_fname.endswith(".LIF"):
            pat, _ = autoguess_life_file(lifep_dir + pat_fname)
            if not verify_baseline(pat, 64):
                print(pat_fname, 'error')


def test_baseline():
    verify_baseline(test_pattern, 64)


def test_construct():
    node = construct(test_pattern)
    validate_tree(node)


def test_centre():
    node = construct(test_pattern)
    for i in range(5):
        old_node = node
        node = centre(node)
        assert node.k == old_node.k + 1
        assert node.n == old_node.n
        ctr_node = inner(node)
        assert ctr_node == old_node


def test_pad():
    node = construct(test_pattern)
    node = pad(node)
    assert is_padded(node)
    node = crop(node)
    assert not is_padded(node)


@lru_cache(None)
def validate_tree(node):
    if node.k > 0:
        assert node.n >= 0 and node.n <= 2 ** (node.k * 2)
        assert node.a.k == node.b.k == node.c.k == node.b.k == node.k - 1
        assert node.n == node.a.n + node.b.n + node.c.n + node.d.n
        assert type(node).__name__ == "Node"
        assert type(node.a).__name__ == "Node"
        assert type(node.b).__name__ == "Node"
        assert type(node.c).__name__ == "Node"
        assert type(node.d).__name__ == "Node"
        validate_tree(node.a)
        validate_tree(node.b)
        validate_tree(node.c)
        validate_tree(node.d)

if __name__ == "__main__":

    test_all_patterns()
    # TODO verify erros:
    # - ACORN.LIF
    # - BHEPTO.LIF
    # - PI.LIF
    # - RPENTO.LIF
    # test_baseline()


