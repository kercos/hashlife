def test_torch_ca_moritztng():
    # from https://github.com/moritztng/cellular/
    import time, threading, torch
    from queue import Queue
    import torch
    import torch.nn.functional as torch_functions

    class Universe:
        def __init__(self, name, state, rule, colors):
            self.name = name
            self.state = state
            self.rule = rule
            self.colors = colors

        def step(self):
            self.state = self.rule(self.state)
    class GameOfLife:
        def __init__(self, device):
            self.parameters = torch.zeros((2, 2, 3, 3), dtype=torch.float32, device=device)
            self.parameters[1, 1, :, :] = 1
            self.parameters[1, 1, 1, 1] = 9

        def __call__(self, state):
            next_state = torch_functions.pad(state, (1, 1, 1, 1), mode="circular")
            next_state = torch_functions.conv2d(next_state, self.parameters)
            next_state = ((next_state == 3) + (next_state == 11) + (next_state == 12)).to(torch.float32)
            next_state[:, 0, :, :] = 1 - next_state[:, 1, :, :]
            return next_state

    def run_universe(stop_event, universe, universe_frequency, device, input_queue):
        while not stop_event.is_set():
            while not input_queue.empty():
                input = input_queue.get()
                top = max(input[0] - input[2], 0)
                bottom = min(input[0] + input[2] + 1, universe[0].state.size(2))
                left = max(input[1] - input[2], 0)
                right = min(input[1] + input[2] + 1, universe[0].state.size(3))
                one_hot = torch.zeros(universe[0].state.size(1), dtype=torch.float32, device=device)
                one_hot[input[3]] = 1
                universe[0].state[0, :, top:bottom, left:right] = one_hot[:, None, None]
            universe[0].step()
            time.sleep(1 / universe_frequency)

    universes = {
        "game_of_life": {
            "rule": GameOfLife,
            "state_colors": [[0, 0, 0], [0, 255, 0]]
        }
    }

    run_universe(
        stop_event = threading.Event(),
        universe = [universes["game_of_life"]()],
        universe_frequency = 30,
        device = 'cpu',
        input_queue = Queue()
    )

if __name__ == "__main__":
    test_torch_ca_moritztng()