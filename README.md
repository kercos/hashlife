<img src="imgs/header.png" width="50%">

Implementation of Gosper's hashlife algorithm. See [johnhw.github.io/hashlife](https://johnhw.github.io/hashlife/index.md.html) for a full explanation.

Usage:

```python
from hashlife import construct, advance, expand
from lifeparsers import autoguess_life_file
from render import render_img

pat, _ = autoguess_life_file("lifep/gun30.lif")
node = construct(pat) # create quadtree
node_30 = advance(node, 30) # forward 30 generations
pts = expand(node_30) # convert to point list
render_img(pts) # render as image
```

<img src="imgs/gun30_30.png">

## Credits

Life patterns in `lifep/` collected by Alan Hensel.

## File Formats
See https://conwaylife.com/wiki/File_formats
- Run Length Encoded (RLE)	ASCII format most commonly-used for storing patterns. It is more cryptic than some other file formats such as plaintext and Life 1.06, but is still quite readable. RLE files are typically saved with a .rle file extension.
- Life 1.05	An ASCII format for storing patterns by simply using dots (.) to represent dead cells and asterisks (*) to represent alive cells. This file format was designed to be easily ported; you can look at a pattern saved in this format in a text editor and figure out what it is. Life 1.05 files are saved with a .lif or .life file extension.
- Life 1.06	An ASCII format that is just a list of coordinates of alive cells. Life 1.06 was designed to be easy and quick for a Life program to read and write, with the drawback being that the file size is very large for large patterns. Life 1.06 files are saved with a .lif or .life file extension.
