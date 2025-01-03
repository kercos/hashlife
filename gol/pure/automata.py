import time
import numpy as np
from numpy.fft import fft2 as np_fft2, ifft2 as np_ifft2
from matplotlib import pyplot as plt, animation
import torch
from torch.fft import fft2 as torch_fft2, ifft2 as torch_ifft2
import scipy
from PIL import Image, ImageDraw
from tqdm import tqdm

class Automata:
    '''
    shape: must be 2d and power of 2 to make things efficient
    board: initial configuration (binary)
    neighborhood: who are my neighbors (maked with 1s)
    torus: rolling over the boundaries (https://en.wikipedia.org/wiki/Torus)
    rule[0]: '1->1' based on "1" neighbours (can't contain 0)
    rule[1]: '0->1' based on "1" neighbours (can't contain 0)
    torch_device: None or in ["cpu", "cuda", "mps"]
    '''
    def __init__(
            self, board, neighborhood, rule, torus=True,
            use_fft = False, # otherwise conv2d
            use_poly_update = False, # life_update, life_update_torch
            torch_device = None,
    ):

        assert (
            0 not in rule[0]
            and
            0 not in rule[1]
        ), "Rule cannot contain zeros"

        assert (
            torch_device in [None, "cpu", "cuda", "mps"]
        ), f"torch_device ({torch_device}) not recognized"

        # board - the grid
        self.board = board

        # neighborhood (e.g, 3x3 in GoL)
        self.neighborhood = neighborhood

        self.size, _, = self.height, self.width = self.shape = self.board.shape

        assert self.height == self.width

        self.minus_size_half_plus_one =  - int(self.size / 2) + 1

        self.rule = rule
        self.torus = torus

        # torch
        self.torch_device = torch_device
        self.use_torch = torch_device is not None

        self.use_poly_update = use_poly_update

        self.use_fft = use_fft
        if self.use_fft:
            nh, nw = self.neighborhood.shape # say (3,3) for Conway's GoL

            # create the FFT kernal (init as zero)
            # put neighborhood mask in the middle (everything else is 0.)
            # TODO check:
            # - not always perfectly centered
            # - e.g., with 3x3 neighborhood and grid size being even (power of two)
            self.kernal = np.zeros(self.shape, dtype=np.float32) # kernal has same shape of board, say (256,256)
            self.kernal[
                (self.shape[0] - nh - 1) // 2 : (self.shape[0] + nh) // 2,
                (self.shape[1] - nw - 1) // 2 : (self.shape[1] + nw) // 2
            ] = self.neighborhood

        if self.use_torch:
            # need to convert arrays to tensor when using torch
            # print('cuda available:', torch.cuda.is_available())
            # print('mps available:', torch.backends.mps.is_available())
            torch_device = torch.device(torch_device)
            torch.set_default_device(torch_device)

            if self.use_fft:
                # less efficient (buth worth trying)
                self.torch_conv_func = self.torch_conv_fft
                self.board = torch.from_numpy(self.board).to(torch_device)
                self.rule = [torch.IntTensor(r).to(torch_device) for r in self.rule] # int
                self.kernal = torch.from_numpy(self.kernal).to(torch_device)
                self.kernal_ft = torch_fft2(self.kernal)
                self.rule_dtype = torch.int32
            else:
                # use conv2d (more efficient) - TODO: check why only works on floats
                # need to convert each rule to float tensor
                self.torch_conv_func = self.torch_conv_conv2d
                self.board = torch.from_numpy(self.board).float().to(torch_device)
                self.rule = [torch.FloatTensor(r).to(torch_device) for r in self.rule] # float
                self.neighborhood = torch.from_numpy(self.neighborhood).float().to(torch_device)
                self.rule_dtype = torch.float32

                # the conv2d class
                self.conv2d_model = torch.nn.Conv2d(
                    in_channels = 1,
                    out_channels = 1,
                    kernel_size = self.neighborhood.shape, # 3x3
                    padding = 'same',
                    padding_mode = 'circular' if self.torus else 'zeros'
                )
                # set the conv2d weight
                with torch.no_grad():
                    self.conv2d_model.weight[0,0] = self.neighborhood
        else:
            # numpy
            if self.use_fft:
                # less efficient (buth worth mentioning)
                self.numpy_conv_func = self.np_conv_fft
                self.kernal_ft = np_fft2(self.kernal) # same shape but floating numbers
            else:
                # use conv2d (more efficient)
                self.numpy_conv_func = self.np_conv_conv2d_scipy_signal_convolve2d

                # TODO: fix me - currently not working
                # self.numpy_conv_func = self.np_conv_conv2d_scipy_ndimage_convolve

                self.np_conv2d_boundary = 'circular' if self.torus else 'fill'


    def get_board_numpy(self, change_to_bool=False, change_to_int=False):
        if self.use_torch:
            result = self.board.cpu().detach().numpy()
        else:
            result = self.board.copy()
        if change_to_bool:
            result = result.astype(bool)
        elif change_to_int:
            result = result.astype(int)
        return result

    def set_board(self, board):
        if self.use_torch:
            if self.use_fft:
                self.board = torch.from_numpy(board).to(self.torch_device)
            else:
                self.board = torch.from_numpy(board).float().to(self.torch_device)
        else:
            self.board = board

    def set_random_board(self):
        if self.use_torch:
            board = torch.rand(self.shape)
            board = board < 0.5 # assume density 0.5
        else:
            board = np.random.uniform(0, 1, self.shape)
            board = board < 0.5 # assume density 0.5

    def get_board_pts(self, only_alive=True):
        from gol.utils import get_board_pts
        return get_board_pts(self.board, only_alive=only_alive)

    '''
    Main 1-step operation (numpy-fft)
    Based on numpy and fft
    '''
    def np_conv_fft(self):

        # fft2 same shape but floating numbers
        # you get torus=True for free
        board_ft = np_fft2(self.board)

        # inverted fft2 (complex numbers)
        # you get torus=True for free
        inverted = np_ifft2(board_ft * self.kernal_ft)

        # get the real part of the complex argument (real number)
        count_real = np.real(inverted)

        # round real part to closest integer
        counts_int = np.rint(count_real)

        # rolling over the boundaries
        # see https://en.wikipedia.org/wiki/Torus
        if not self.torus:
            pass
            # TODO: fix me
            # by default fft assumes torus boundaries condition
            # need to implment this if we want strict boundaries

        # this is necessary (only in fft) for making things work
        # TODO: understand this better
        counts_int = self.np_recenter_conv(counts_int)

        return counts_int

    '''
    Main 1-step operation (numpy)
    Based on numpy and convolve2d from scipy.signal
    '''
    def np_conv_conv2d_scipy_signal_convolve2d(self):

        # the conv2d step (via scipy)
        counts_int = scipy.signal.convolve2d(
            self.board,
            self.neighborhood,
            mode = 'same',
            boundary = self.np_conv2d_boundary # 'circular' if torus, 'fill' if strict
            # rolling over the boundaries
            # see https://en.wikipedia.org/wiki/Torus
        )

        # round real part to closest integer
        # TODO: double check type, this may not be necessary
        counts_int = np.rint(counts_int)

        return counts_int

    '''
    Main 1-step operation (numpy-conv-scipy)
    Based on scipy.ndimage.convolve
    TODO: fix me - currently not working
    '''
    def np_conv_conv2d_scipy_ndimage_convolve(self):
        ndimage_gol_k = np.array(
            [
                1,1,1,
                1,2,1,
                1,1,1
            ]
        ).reshape(3,3)
        result = scipy.ndimage.convolve(
            self.board,
            ndimage_gol_k,
            mode="wrap"
        )
        return (result>4) & (result<8)


    '''
    Rolling over the boundaries (numpy version)
    see https://en.wikipedia.org/wiki/Torus
    '''
    def np_recenter_conv(self, counts_int):
        return np.roll(
            counts_int,
            self.minus_size_half_plus_one,
            axis=(0,1)
        )

    '''
    Apply the rule (numpy version)
    '''
    def np_apply_rule(self, count_ones_neighbours):

        board_ones = self.board == 1
        board_zeros = ~ board_ones # negation

        new_board = np.zeros(self.shape)

        # rule[0] (survival): '1->1' based on count of "1" neighbours
        new_board[
            np.where(
                np.isin(count_ones_neighbours, self.rule[0]).reshape(self.shape)
                &
                board_ones # on cells
            )
        ] = 1

        # rule[1] (reproduction): '0->1' based on count of "1" neighbours
        new_board[
            np.where(
                np.isin(count_ones_neighbours, self.rule[1]).reshape(self.shape)
                &
                board_zeros # off cells
            )
        ] = 1

        # all other cells stay zeros (1->0, 0->0)

        self.board = new_board

    '''
    Step update function using numpy
    '''
    def np_update_board(self):

        # counting number of alive cells in neighbourhood (same shape)
        # via fft or conv2 based on torch_conv_func
        # (np_conv_fft or np_conv_conv2d_scipy_signal_convolve2d)
        count_ones_neighbours = self.numpy_conv_func()

        # apply rule (update board inplace)
        self.np_apply_rule(count_ones_neighbours)

    '''
    Polynomial life update via numpy
    Credits to Raymond Baranski (Alex)
    '''
    def np_update_board_poly(self):
        counts = self.numpy_conv_func()
        self.board = (
            # np.sign(-1*(1-x)*(y-2.5)*(y-3.5)-x*(y-1.5)*(y-3.5))
            np.sign(
                -counts ** 2
                - self.board * counts
                + 3.5 * self.board
                + 6 * counts
                - 8.75
            )
            +1
        ) /2

    '''
    Main 1-step operation (torch version)
    Usign FFT (not as efficient as torch.nn.functional.conv2d)
    '''
    def torch_conv_fft(self):

        # fft2 same shape but floating numbers
        board_ft = torch_fft2(self.board)

        # inverted fft2 (complex numbers)
        inverted = torch_ifft2(board_ft * self.kernal_ft)

        # get the real part of the complex argument (real number)
        count_real = torch.real(inverted)

        # round real part to closest integer
        counts_int = torch.round(count_real).int()

        if not self.torus:
            pass
            # TODO: fix me
            # by default fft assumes torus boundaries condition
            # need to implment this if we want strict boundaries

        # this is necessary (only in fft) for making things work
        # TODO: understand this better
        counts_int = self.torch_recenter_conv(counts_int)

        return counts_int

    '''
    Main 1-step operation (torch)
    torch version using conv2d (more efficient)
    '''
    def torch_conv_conv2d(self):

        # create two more dims
        board_conv = self.board[None,None,:,:]

        # apply convolution step
        counts_int = self.conv2d_model(board_conv)

        # taking only first two channels in first two dim
        counts_int = counts_int[0,0,:,:]

        # round real part to closest integer
        # make sure its dtype is self.rule_dtype
        counts_int = torch.round(counts_int)

        return counts_int

    '''
    Rolling over the boundaries (torch version)
    see https://en.wikipedia.org/wiki/Torus
    '''
    def torch_recenter_conv(self, counts_int):
        return torch.roll(
            counts_int,
            (
                self.minus_size_half_plus_one,
                self.minus_size_half_plus_one
            ), # need tuple here because torch behaves differently from numpy apparently
            dims=(0,1)
        )

    '''
    Apply the rule (torch version)
    '''
    def torch_apply_rule(self, count_ones_neighbours):
        board_ones = self.board == 1
        board_zeros = ~ board_ones # negation

        new_board = torch.zeros(self.shape, dtype=self.rule_dtype) # int or float

        # rule[0] (survival): '1->1' based on count of "1" neighbours
        new_board[
            torch.where(
                torch.isin(count_ones_neighbours, self.rule[0]).reshape(self.shape)
                &
                board_ones # on cells
            )
        ] = 1

        # rule[1] (reproduction): '0->1' based on count of "1" neighbours
        new_board[
            torch.where(
                torch.isin(count_ones_neighbours, self.rule[1]).reshape(self.shape)
                &
                board_zeros # off cells
            )
        ] = 1

        # all other cells stay zeros (1->0, 0->0)

        self.board = new_board

    '''
    Step update function using torch
    '''
    def torch_update_board(self):

        # counting number of alive cells in neighbourhood (same shape)
        # via fft or conv2 based on torch_conv_func
        # (torch_conv_fft or torch_conv_conv2d)
        count_ones_neighbours = self.torch_conv_func()

        # apply rule (update board inplace)
        self.torch_apply_rule(count_ones_neighbours)

    '''
    Step update function using torch and poly
    Credits to Raymond Baranski (Alex)
    '''
    def torch_update_board_poly(self):
        counts = self.torch_conv_func()
        self.board = (
            # np.sign(-1*(1-x)*(y-2.5)*(y-3.5)-x*(y-1.5)*(y-3.5))
            torch.sign(
                -counts ** 2
                - self.board * counts
                + 3.5 * self.board
                + 6 * counts
                - 8.75
            )
            +1
        ) /2

    '''
    Step update function (general)
    '''
    def update_board(self):
        if self.use_torch:
            if self.use_poly_update:
                self.torch_update_board_poly()
            else:
                self.torch_update_board()
        else:
            if self.use_poly_update:
                self.np_update_board_poly()
            else:
                self.np_update_board()

    '''
    Multi-Step update function (general)
    '''
    def advance(self,iterations=1):
        if self.use_torch:
            if self.use_poly_update:
                for _ in range(iterations):
                    self.torch_update_board_poly()
            else:
                for _ in range(iterations):
                    self.torch_update_board()
        else:
            # use numpy
            if self.use_poly_update:
                for _ in range(iterations):
                    self.np_update_board_poly()
            else:
                for _ in range(iterations):
                    self.np_update_board()

    '''
    Main Benchmark to run it as fast as possible (without visualizing it)
    '''
    def benchmark(self, iterations):
        # TODO: change to perf_counter() or pytorch profiler ?
        start = time.process_time()

        self.advance(iterations)

        # TODO: change to perf_counter() or pytorch profiler ?
        ellapsed = time.process_time() - start

        hz = iterations / ellapsed

        hz_B_cell = hz * self.size * self.size / 10 ** 9 # Billions

        device_str = f'Torch-{self.torch_device}' if self.torch_device else 'Numpy'
        conv2d_fft_poly_str = 'FFT' if self.use_fft else 'Conv2D'
        if self.use_poly_update:
            conv2d_fft_poly_str += '-POLY'
        shape_str = f'{self.size}x{self.size}'

        print(
            f'Device: {device_str}\n',
            f'Benchmark {conv2d_fft_poly_str}:',
            f'{shape_str} grid,',
            f'{iterations} iters,',
            f'torus={self.torus}\n',
            f'{hz:.0f} Hz (board)\n',
            f'{ellapsed:.1f} s (cells)\n',
            f'{hz_B_cell:.2f} BHz (per cell)'
        )

    '''
    Show grid in real time (rely on numpy for now)
    '''
    def animate(
            self,
            iterations=None,
            name='animation',
            interval=0,
            progress=True):

        self.animate_iter = 0

        if progress:
            bar = tqdm() # total?

        def update_animation(*args):
            if self.animate_iter == iterations:
                return None
            self.animate_iter += 1

            if progress:
                bar.update()

            # update board once (using numpy or torch)
            self.update_board()

            # make sure it's numpy array
            board_numpy = self.get_board_numpy()
            self.image.set_array(board_numpy)
            return self.image

        fig = plt.figure(name)
        plt.axis("off")

        # first frame
        self.image = plt.imshow(
            self.get_board_numpy(),
            interpolation="nearest",
            cmap=plt.cm.gray
        )

        _ = animation.FuncAnimation(
            fig,
            update_animation,
            interval=interval, # ms
            cache_frame_data=False # or save_count=MAX_FRAMES
        )

        # always force show
        plt.show()

    def save_last_frame(self, filename, grid=False):

        # make sure it's numpy array
        board_np = self.get_board_numpy()

        if filename.endswith('npy'):
            np.save(filename, board_np)
        else:
            wh_src = board_np.shape[0] # original width, height
            px_wh = 50 # pixel width, height
            wh = wh_src * px_wh # final width, height

            # There seem to be issues when using mode 1 with numpy arrays.
            # see https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-from-array-with-mode-1
            # board_np = board_np.astype(np.uint8)
            # img = Image.fromarray(board_np, mode='L').convert('1')

            # gray image
            board_np = board_np.astype(np.uint8) * 255
            # board_np = 255 * np.stack((board_np,) * 3, axis=-1) # RGB
            img = Image.fromarray(board_np, mode='L')

            # set size
            # img = img.resize((wh, wh), Image.Resampling.BOX)

            # draw lines
            # TODO: fix me
            if grid:
                board_np = board_np.astype(np.uint8)
                line_color = (255,100,0)
                palette = [(0,0,0), (255,255,255), line_color]
                img.putpalette(palette)
                draw = ImageDraw.Draw(img)
                for i in range(1,wh_src):
                    # (0,0) at top-left corner
                    xy = i * px_wh
                    shape_v = [(xy, 0), (xy, wh)]
                    shape_h = [(0, xy), (wh, xy)]
                    draw.line(shape_v, fill=line_color, width=2)
                    draw.line(shape_h, fill=line_color, width=2)

            img.save(filename)

    def show_current_frame(self, name, force_show=True):
        if self.use_torch:
            self.board = self.board.cpu().detach().numpy()

        plt.figure(name, figsize=(5, 5))
        plt.imshow(
            self.board,
            interpolation="nearest",
            cmap=plt.cm.gray
        )
        plt.axis("off")

        if force_show:
            plt.show()

    '''
    Get the period of the cycle (for torus boards)
    '''
    def get_cycle_period(self, advance_gen=100, max_period=1000):
        if advance_gen:
            self.advance(advance_gen)
        if self.use_torch:
            first_board = self.board.detach().clone()
            for p in range(1,max_period+1):
                self.advance()
                if torch.equal(first_board, self.board):
                    return p
            return None # no cycle found (up to max_period)
        else:
            first_board = self.board.copy()
            for p in range(1,max_period+1):
                self.advance()
                if np.all(first_board==self.board):
                    return p
            return None # no cycle found (up to max_period)
