import time
from hl.lifeparsers import autoguess_life_file
from hl.hashlife import (
    construct, ffwd, successor, join,
    expand, advance, centre
)

# run the breeder forward many generations
def load_lif(fname):
    pat, comments = autoguess_life_file(fname)
    return construct(pat)


if __name__ == "__main__":
    init_t = time.perf_counter()
    print(ffwd(load_lif("lifep/breeder.lif"), 64))
    t = time.perf_counter() - init_t
    print(f"Computation took {t*1000.0:.1f}ms")
    print(successor.cache_info())
    print(join.cache_info())

    from render import render_img
    import matplotlib.pyplot as plt

    ## test the Gosper glider gun
    pat = load_lif("lifep/gun30.LIF")
    pat = load_lif("lifep/gun30.lif")
    render_img(expand(pat))
    plt.savefig("imgs/gun30_0.png", bbox_inches="tight")
    render_img(expand(advance(centre(centre(pat)), 30)))
    plt.savefig("imgs/gun30_30.png", bbox_inches="tight")

    render_img(expand(advance(centre(centre(pat)), 120), level=0))
    plt.savefig("imgs/gun30_120_0.png", bbox_inches="tight")
    render_img(expand(advance(centre(centre(pat)), 120), level=1))
    plt.savefig("imgs/gun30_120_1.png", bbox_inches="tight")
    render_img(expand(advance(centre(centre(pat)), 120), level=2))
    plt.savefig("imgs/gun30_120_2.png", bbox_inches="tight")
    render_img(expand(advance(centre(centre(pat)), 120), level=3))
    plt.savefig("imgs/gun30_120_3.png", bbox_inches="tight")
