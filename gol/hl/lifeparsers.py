"""
A collection of routines to load Life from various common formats.

Supported formats include:
* Al Hensel's Life 1.05
* Al Hensel's Life 1.06
* David Bell's dblife
* XLife (simple patterns only, which is most of them)
* RLE

Includes a useful wrapper function which will ID a file and open it
    autoguess_life_file()

Return format from loaders is a list of (x,y) tuples, without
any adjustment of the coordinates from the original file.

to_numpy() can convert this to a dense binary array, with the origin
moved to (0,0).

Writing RLE files is also supported with to_rle() and write_rle()

"""
import os, re, textwrap
import numpy as np

# Convert a list of (x,y) positions to a NumPy array
def to_numpy(pos):
    pos = np.array(pos)
    print(pos.shape)
    pos[:, 0] -= np.min(pos[:, 0])
    pos[:, 1] -= np.min(pos[:, 1])
    result = np.zeros((np.max(pos[:, 1]) + 1, np.max(pos[:, 0]) + 1))
    for x, y in pos:
        result[y, x] = 1
    return result

def to_rle(pts, fixed_size=None):
    """Convert a point list to RLE format.
    Pts is a list of ON points: (x,y) coordinates
    If crop is True, it outputs only the region of minimum region of where cells are on
    Returns:
        tuple (rle, (width, height))
        rle: the RLE string,
        width, height: bounds of the pattern """

    # sort by x, then y
    pts.sort(key=lambda p: p[0])
    max_x = pts[-1][0]
    min_x = pts[0][0]
    pts.sort(key=lambda p: p[1])
    max_y = pts[-1][1]
    min_y = pts[0][1]

    crop = fixed_size is None

    if not crop:
        min_x, min_y = 0, 0
        max_x, max_y = fixed_size-1, fixed_size-1

    line = 0
    column = 0
    count_on_off = [0,0]
    out = []

    def append_count_symbol(count, symbol):
        if count==0:
            return False
        if count==1:
            out.append(symbol)
            return True
        else: # > 1
            out.append(f"{count}{symbol}")
            return True

    # write out the on cells
    def flush_symbol(on):
        symbol_index = 0 if on else 1
        count = count_on_off[symbol_index]
        symbol = 'o' if on else 'b'
        if append_count_symbol(count, symbol):
            # reset counter
            count_on_off[symbol_index] = 0

    for pt in pts:
        if crop:
            pt_x, pt_y = (pt[0] - min_x, pt[1] - min_y)
        else:
            pt_x, pt_y = pt[0], pt[1]

        # y co-ord change, write out new lines
        if pt_y != line:
            flush_symbol(on=True)
            count_lines = pt_y - line
            if count_lines > 0:
                append_count_symbol(count_lines, '$')
            line = pt_y
            column = 0

        # mark blanks (skipped in pts)
        # count_on_off[1] = 0 # count_off (reset) - TODO: needed?
        assert count_on_off[1] == 0
        while column != pt_x:
            column += 1
            # count-off increment
            count_on_off[1] += 1 #
        if count_on_off[1] > 0: # count_off
            # write out pending on/off cells
            flush_symbol(on=True)
            flush_symbol(on=False)
        # count-on increment
        count_on_off[0] += 1
        column += 1


    # write out pending on/off cells
    flush_symbol(on=True)
    flush_symbol(on=False)

    out.append("!")
    rle_str = "".join(out)
    size_x = max_x - min_x + 1
    size_y = max_y - min_y + 1
    return rle_str, (size_x, size_y)


def write_rle(filepath, pts, fixed_size=None, torus=True, rule='B3/S23', comments=[]):
    """Write a point list to a file, with an optional comment block"""
    rle, (x, y) = to_rle(pts, fixed_size=fixed_size)

    lines = []

    # size header
    if torus:
        header = f'x = {x}, y = {y}, rule = {rule}:T{x},{y}\n'
    else:
        header = f'x = {x}, y = {y}, rule = {rule}\n'
    lines.append(header)
    # comments
    for comment in comments:
        lines.append(f"#C {comment}\n")

    # rle, 70 char max width
    rle = textwrap.fill(rle, 70)
    lines.append(rle)

    # write to file
    if filepath is not None:
        with open(filepath, 'w') as fout:
            fout.writelines(lines)

    return lines


def rle_string(pts, comments=[]):
    """Write a point list to a file, with an optional comment block"""
    rle, (x, y) = to_rle(pts)
    output = []

    # size header
    output.append("x = %d, y = %d\n" % (x, y))
    # comments
    for comment in comments:
        output.append("#C %s" % comment.strip())

    output.append("\n")
    # rle, 70 char max width
    rle = textwrap.fill(rle, 70)
    output.append(rle)
    return "\n".join(output)


def parse_life_105(file):
    """Parse a Life 1.05 file,  returning a tuple:
        positions: list of (x,y) co-ordinates
        comments: all comments in file, as a list of strings, one per line.
    """
    lines = file.split("\n")
    comments = []
    positions = []
    ox, oy = 0, 0
    x, y = ox, oy

    pattern_105 = r"\s*(\.|\*|o|O)+\s*\Z"
    for line in lines:
        line = line.strip().rstrip()

        if line.startswith("#"):
            # comment
            if line[1] in "CcDd":
                comments.append(line[2:])
            # new block definition
            if line[1] in "Pp":
                coords = line[2:]
                try:
                    ox, oy = [int(p) for p in coords.split()]
                    x, y = ox, oy
                except:
                    pass
        else:
            # skip blanks
            if len(line) > 0 and re.match(pattern_105, line):
                # only fill in points which are active
                for char in line:
                    if char == "*" or char == "o" or char == "O":
                        positions.append((x, y))
                    x += 1
                y = y + 1
                x = ox
    comments = "\n".join(comments)

    return positions, comments

def parse_life_106(file):
    """
     Parse a Life 1.06 file, returning a tuple:
        positions: list of (x,y) co-ordinates of ON cells
        comments: all comments in file, as a list of strings, one per line
    """
    lines = file.split("\n")
    comments = []
    positions = []

    pattern_106 = r"\s*\-?[0-9]+\s+\-?[0-9]+\s*"
    for line in lines:
        line = line.strip().rstrip()

        if line.startswith("#"):
            # strip out comments
            if line[1] in "CcDdnN":
                comments.append(line[2:])
        else:

            if re.match(pattern_106, line):
                try:
                    x, y = [int(p) for p in line.split()]
                    positions.append((x, y))
                except:
                    pass
    comments = "\n".join(comments)
    return positions, comments

def parse_dblife(file):
    """Parse an DBLife file, returning a tuple:
        positions: list of (x,y) co-ordinates
        comments: all comments in file, as a list of strings, one per line.
    """
    lines = file.split("\n")
    comments = []
    positions = []
    x = 0
    y = 0
    dblife_pattern = r"((\d*)(\.|O|o|\*))*"

    for line in lines:
        line = line.strip().rstrip()

        if line.startswith("!"):
            comments.append(line[2:])

        # check if this is part of the pattern
        if re.match(dblife_pattern, line):

            count = 0
            for char in line:

                # repeat counts
                if char.isdigit():
                    count *= 10
                    count += int(char)

                # blanks
                if char in ".":
                    if count != 0:
                        x += int(count)
                    else:
                        x += 1
                    count = 0
                # ons
                if char in "oO*":
                    if count != 0:
                        for i in range(count):
                            positions.append((x, y))
                            x += 1
                    else:
                        positions.append((x, y))
                        x += 1
                    count = 0
                count = 0

            # newlines
            y += 1
            x = 0
            count = 0

    return positions, comments


def parse_rle(rle):
    """Parse an RLE string, returning a tuple:
        positions: list of (x,y) co-ordinates"""
    lines = rle.split("\n")
    comments = []
    positions = []
    x = 0
    y = 0

    complete = False
    for line in lines:
        line = line.strip().rstrip()
        if len(line) == 0:
            pass
        elif complete:
            comments.append(line)

        elif line.startswith("#"):
            # extract comment/owner
            if complete or line[1] in "cCoOnN":
                comments.append(line[2:])
            # get offsets
            if line[1] in "pP":
                coords = line[2:]
                try:
                    x, y = [int(p) for p in coords.split()]
                except:
                    pass

        # skip any size line -- we don't need it
        elif line.startswith("x"):
            continue
        else:
            count = 0
            for char in line:

                # repeat counts
                if char.isdigit():
                    count *= 10
                    count += int(char)

                # blanks
                if char in "bB":
                    if count != 0:
                        x += int(count)
                    else:
                        x += 1
                    count = 0

                # ons
                if char in "oO":
                    if count != 0:
                        for i in range(count):
                            positions.append((x, y))
                            x += 1
                    else:
                        positions.append((x, y))
                        x += 1
                    count = 0

                # newlines
                if char in "$":
                    if count != 0:
                        y += int(count)
                    else:
                        y += 1
                    x = 0
                    count = 0
                if char in "!":
                    complete = True
                    break

    return positions, comments


def autoguess_life_file(fname):
    """Open the given file, try and identify the file type
    and return the parsed version of the file. Supports:
    * Life 1.05
    * Life 1.06
    * DBLife
    * XLife
    * RLE
    """
    base, ext = os.path.splitext(fname)
    f = open(fname)
    text = f.read()
    f.close()
    lines = text.split("\n")

    first_line = lines[0].strip().rstrip()

    # life 1.05
    if first_line.startswith("#Life 1.05"):
        return parse_life_105(text)
    if first_line.startswith("#Life 1.06"):
        return parse_life_106(text)
    elif first_line.startswith("!"):
        return parse_dblife(text)

    # ok, now it could be an RLE file, or it could be an XLIFE file
    rle_result = parse_rle(text)
    result_105 = parse_life_105(text)
    result_106 = parse_life_106(text)

    r1 = len(rle_result[0])
    r2 = len(result_105[0])
    r3 = len(result_106[0])

    # rle gave most cells
    if r1 > r2 and r1 > r3:
        print("Guessed RLE")
        return rle_result

    if r2 > r1 and r2 > r3:
        print("Guessed Life 1.05")
        return result_105

    if r3 > r1 and r3 > r1:
        print("Guessed Life 1.06")
        return result_106

    # default, RLE
    return rle_result


def read_rle(fname):
    """Open and parse an RLE file"""
    f = open(fname)
    positions, comments = parse_rle(f.read())
    f.close()
    return positions, comments


if __name__ == "__main__":
    import sys

    pat, comments = autoguess_life_file(sys.argv[1])
    print((rle_string(pat, comments=comments.split("\n"))))

