# Emergence of Structure in 2D Cellular Automata (Game of Life)

Explore emerging patterns in GoL, something along [these lines](https://softologyblog.wordpress.com/2019/09/03/automatic-detection-of-interesting-cellular-automata/) - not clear if it has been investigated thoroughly so far.

## Setup
- activate env
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- install requirements (after env is activated)
  ```
  pip install -r requirements.txt
  ```
- you may need PyQt5 in addition to visualize animation (e.g., under Ubuntu)
  ```
  pip install PyQt5
  ```

## Run the code
- main pure
  ```
  python -m gol.main_pure
  ```
- main cycles
  ```
  python -m gol.main_cycles
  ```
- main hl (hashlife)
  ```
  python -m gol.main_hl
  ```
- benchmark: pure, golly (draft)
  ```
  python -m gol.benchmark
  ```
- test hl/pure consistency
  ```
  python -m gol.test_hl_pure
  ```
- test hl jumps
  ```
  python -m gol.test_hl_jumps
  ```
- test golly
  ```
  python -m gol.test_golly
  ```
- process lexicon (`input/lex_asc` below)
  ```
  python -m gol.process_lexicon
  ```
  TODO: see also https://conwaylife.com/wiki

## Project Structure
- `pure`: implementation with efficient `numpy`/`pytorch` via `fft` and `conv2d` derived from [njbbaer](https://gist.github.com/njbbaer/4da02e2960636d349e9bae7ae43c213c).
- `hl`: hashlife from [johnhw](https://github.com/johnhw/hashlife).
- `moritztng`: copy of [moritztng](https://github.com/moritztng/cellular) (2d conv and web visualizer)

## input/output dirs (gitignored)
- `input`
  - `lifep`: from https://github.com/johnhw/hashlife/tree/master/lifep
  - `lex_asc`: from https://conwaylife.com/ref/lexicon/lex_home.htm
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
- see other life catalogues
  - https://catagolue.hatsya.com/census
  - https://conwaylife.com/wiki/Main_Page
- see `moritztng` (or remove if not relevant)

## CONTACTS
[@kercos](https://t.me/kercos)