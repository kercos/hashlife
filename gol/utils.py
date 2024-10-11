import numpy as np

def init_gol_board_neighborhood_rule(
        shape_x = 16,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
    ):

    shape = (shape_x, shape_x)

    if initial_state == 'random':
        # initialize random generator
        rng = np.random.default_rng(seed)
        board = rng.uniform(0, 1, shape)
        board = board < density
    elif initial_state == 'square':
        sq = 2 # alive square size in the middle of the board
        assert sq % 2 == 0
        board = np.zeros(shape)
        board[
            shape_x//2-sq//2:shape_x//2+sq//2,
            shape_x//2-sq//2:shape_x//2+sq//2
        ] = 1 # alive
    else:
        assert initial_state.endswith('.npy')
        board = np.load(initial_state)

    neighborhood = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
    )

    # GoL Rule (same as Conway class):
    rule = [
        [2, 3], # 'on->on': (2,3): "on" neighbours (can't contain 0)
        [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
    ]

    # GoL Rule with neighborhood all ones:
    # neighborhood = np.ones((3,3))
    # rule = [
    #     [3, 4], # 'on->on': (2,3): "on" neighbours (can't contain 0)
    #     [3]     # 'off->on': (3,): "on" neighbours (can't contain 0)
    # ]

    # exploring other rules:
    # rule = [
    #     [2], # 'on->on': "on" neighbours (can't contain 0)
    #     [1]  # 'off->on':   "on" neighbours (can't contain 0)
    # ]

    return board, neighborhood, rule

# Convert a (dense) NumPy array to list of (x,y) positions in life 1.06
def numpy_to_life_106(board_np, filepath):
    # see http://www.mirekw.com/ca/ca_files_formats.html
    header = '#Life 1.06'
    shape_x, shape_y = board_np.shape

    lines = [header]
    for x in range(shape_x): # columns (x)
        for y in range(shape_y): #rows (y)
            if board_np[y,x]:
                lines.append(f'{x} {y}') # x,y

    # write to file
    with open(filepath, 'w') as fout:
        # lines with return except for last
        lines_with_return = \
            [f'{l}\n' for l in lines[:-1]] + \
            [lines[-1]]
        fout.writelines(lines_with_return)

if __name__ == "__main__":
    board, neighborhood, rule = init_gol_board_neighborhood_rule(
        shape_x = 16,
        initial_state = 'random', # 'random', 'square', 'filename.npy'
        density = 0.5, # only used on initial_state=='random'
        seed = 123,
    )
    filepath_out = 'output/base16_0.LIFE'
    numpy_to_life_106(board, filepath_out)