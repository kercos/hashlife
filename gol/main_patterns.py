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

if __name__ == "__main__":
    wave()
    # mold()
