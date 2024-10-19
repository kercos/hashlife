# Emergence of Structure in 2D Cellular Automata (Game of Life)

Explore emerging patterns in GoL, something along [these lines](https://softologyblog.wordpress.com/2019/09/03/automatic-detection-of-interesting-cellular-automata/) - not clear if it has been investigated thoroughly so far.

## Run the code
- main (pure)
  ```
  python -m gol.main_pure
  ```
- benchmark: pure, golly (draft)
  ```
  python -m gol.benchmark
  ```
- main (hl)
  ```
  python -m gol.main_hl
  ```
- test (pure/hl)
  ```
  python -m gol.test
  ```

## Project Structure
- `pure`: implementation with efficient `numpy`/`pytorch` via `fft` and `conv2d` derived from [njbbaer](https://gist.github.com/njbbaer/4da02e2960636d349e9bae7ae43c213c).
- `hl`: hashlife from [johnhw](https://github.com/johnhw/hashlife).
- `moritztng`: copy of [moritztng](https://github.com/moritztng/cellular) (2d conv and web visualizer)

## input/output dirs (gitignored)
- `input`
  - `hl_imgs` (hl)
  - `hl_lifep` (hl)
- `output`
  - `manual`: manual tests for `pure`
  - `base`: output for `test` (pure/hl)
  - `hl_imgs`: output fro hl

## See also:
- [ca file formats](http://www.mirekw.com/ca/ca_files_formats.html)
- [johnhw (hashlife)](https://johnhw.github.io/hashlife/index.md.html) for a full explanation.
- [SortaSota (julia)](https://rivesunder.github.io/SortaSota/2021/09/27/faster_life_julia.html)
- [rivesunder (carle)](https://github.com/rivesunder/carle)

## TODO
- see `moritztng` (or remove if not relevant)

## CONTACTS
[@kercos](https://t.me/kercos)