# Emergence of Structure in 2D Cellular Automata (Game of Life)

## Project Structure
- `pure`: standard implementation with efficient `numpy`/`pytorch` implementation via `fft` and `conv2d`. See `main.py` initially derived from [njbbaer](https://gist.github.com/njbbaer/4da02e2960636d349e9bae7ae43c213c).
- `hl`: hashlife from [johnhw](https://github.com/johnhw/hashlife).
- `moritztng`: https://github.com/moritztng/cellular

## input/output dirs (gitignored)
- `input`
  - `hl_imgs` (hl)
  - `hl_lifep` (hl)
- `output`
  - `manual`: manual tests for `pure` (4x4)

## run the code
- pure
  - `python -m gol.pure.main`
- hl
  - `python -m gol.hl.main`

## See also:
- [johnhw (hashlife)](https://johnhw.github.io/hashlife/index.md.html) for a full explanation.
- [SortaSota (julia)](https://rivesunder.github.io/SortaSota/2021/09/27/faster_life_julia.html)
- [rivesunder (carle)](https://github.com/rivesunder/carle)

## TODO
- see `moritztng` (or remove if not relevant)