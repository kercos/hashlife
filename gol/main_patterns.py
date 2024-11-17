# GoL by default
from gol.main_cycles import visualize_cycle

def wave():
    visualize_cycle(
        size = 16,
        init_state = 1840561505,
    )

def mold():
    visualize_cycle(
        size = 16,
        init_state = 407304018,
    )

def four_eyes():
    visualize_cycle(
        size = 16,
        init_state = 1778200793,
        padding = 3
    )

def new():
    visualize_cycle(
        size = 16,
        init_state = 431986027,
        # init_state = 955454776,
        # init_state = 1539012979,
        # init_state = 1390382698,
        # init_state = 580683925,
        # init_state = 734800613,
        # init_state = 1671604753,
        padding = 0
    )

if __name__ == "__main__":
    # four_eyes()
    # wave()
    # mold()
    new()
