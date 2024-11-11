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

if __name__ == "__main__":
    four_eyes()
    # wave()
    # mold()
