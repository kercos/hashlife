import time
from matplotlib import pyplot as plt
from gol.main_pure import Automata
from gol.hl.lifeparsers import write_rle
from gol.base import generate_base

'''
Usage:  bgolly [options] patternfile
 -m --generation    How far to run
 -i --stepsize      Step size
 -M --maxmemory     Max memory to use in megabytes
 -T --maxtime       Max duration
 -b --benchmark     Show timestamps
 -2 --exponential   Use exponentially increasing steps
 -q --quiet         Don't show population; twice, don't show anything
 -r --rule          Life rule to use
 -s --search        Search directory for .rule files
 -h --hashlife      Use Hashlife algorithm
 -a --algorithm     Select algorithm by name
 -o --output        Output file (*.rle, *.mc, *.rle.gz, *.mc.gz)
 -v --verbose       Verbose
 -t --timeline      Use timeline
    --render        Render (benchmarking)
    --progress      Render during progress dialog (debugging)
    --popcount      Popcount (benchmarking)
    --scale         Rendering scale
    --autofit       Autofit before each render
    --exec          Run testing script
'''

def benchmark_golly(size, rle_filepath, iterations):
    import subprocess
    cmd = './golly-4.3-src/bgolly'

    start = time.process_time()

    subprocess.run(
        [
            cmd,
            '-m', str(iterations),# -m --generation    How far to run
            '-q', #  Don't show population;
            '-q', #  twice, don't show anything
            rle_filepath
        ]
    )

    # TODO: change to perf_counter() or pytorch profiler ?
    ellapsed = time.process_time() - start

    hz = iterations / ellapsed

    hz_B_cell = hz * size * size / 10 ** 9 # Billions


    print(
        "Size", size,
        "Iters", iterations,
        f"cells in {ellapsed:.1f} s",
        f"{hz:.0f} Hz (board)",
        f"{hz_B_cell:.2f} BHz (cell)"
    )

def run_golly(rle_filepath, iterations, output_path):
    import subprocess
    cmd = './golly-4.3-src/bgolly'
    subprocess.run(
        [
            cmd,
            '-m', str(iterations),# -m --generation    How far to run
            '-o', output_path, # -o --output (*.rle, *.mc, *.rle.gz, *.mc.gz)
            # '-q', #  Don't show population;
            # '-q', #  twice, don't show anything
            rle_filepath
        ]
    )

def test_golly(size = 16, iterations=100, show=False):

    node, board, neighborhood, rule = generate_base(
        size = size,
        seed=123
    )

    # init automata
    automata = Automata(
        board = board,
        neighborhood = neighborhood,
        rule = rule,
        torus = True,
        use_fft = False,
        torch_device = None,
    )

    def export_automata(it):
        base_name = f'output/golly/base{size}_{it}'
        # get on cells (x,y) coordinates from automata board
        pts = automata.get_board_pts(only_alive=True)
        # write RLE
        write_rle(fixed_size=size, filepath=f'{base_name}.RLE', pts=pts)
        # write PNG
        automata.save_last_frame(filename=f'{base_name}.png')
        if show:
            automata.show_current_frame(
                name = f'it={it}',
                force_show = False
            )

    export_automata(it=0)

    automata.advance(iterations)
    export_automata(iterations)

    if show:
        plt.show()

    run_golly(
        rle_filepath = f'output/golly/base{size}_{iterations}.RLE',
        iterations = iterations,
        output_path = f'output/golly/base{size}_{iterations}_golly.RLE'
    )

    benchmark_golly(
        size,
        rle_filepath = f'output/golly/base{size}_{iterations}.RLE',
        iterations = iterations
    )



if __name__ == "__main__":
    test_golly(16)
