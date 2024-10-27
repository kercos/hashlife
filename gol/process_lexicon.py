'''
see https://conwaylife.com/ref/lexicon/lex_home.htm
you need to download ASCII version in `LEX_ASC_PATH` (see below)
'''

import os
import glob

LEX_ASC_PATH = 'input/lex_asc/lexicon.txt'
LEX_ASC_PATTERNS_DIR = 'input/lex_asc/patterns'
LEX_ASC_PATTERNS_REF = 'input/lex_asc/ref_patterns.txt'
LEX_ASC_NON_PATTERNS = 'input/lex_asc/non_patterns.txt'

PATTERN_COUNTER = 0
REF_COUNTER = 0
NON_PATTERN_COUNTER = 0


def remove_pattern_files():
    files = glob.glob(f'{LEX_ASC_PATTERNS_DIR}/*')
    for f in files:
        os.remove(f)

def count_on_cells(pattern_lines):
    count = 0
    for l in pattern_lines:
        count += l.count('*')
    return count

def process_pattern(block, filesystem_write=False, verbose=True):
    global PATTERN_COUNTER, REF_COUNTER, NON_PATTERN_COUNTER
    pattern_name, pattern = None, None
    first_line = block[0].strip()
    if len(block)==1:
        # reference to other pattern
        # e.g.,
        # :0hd Demonoid:  See {Demonoid}.
        REF_COUNTER += 1
        if filesystem_write:
            with open(LEX_ASC_PATTERNS_REF, 'a') as fout:
                fout.write(first_line)
                fout.write('\n')
    else:
        comment_lines = []
        pattern_lines = []
        for l in block:
            if l == '\n':
                continue
            if l.startswith('\t'):
                pattern_lines.append(l.strip())
            else:
                comment_lines.append(l.strip())

        if len(pattern_lines) == 0 or count_on_cells(pattern_lines)==0:
            # no pattern found in block
            NON_PATTERN_COUNTER += 1
            if filesystem_write:
                with open(LEX_ASC_NON_PATTERNS, 'a') as fout:
                    if NON_PATTERN_COUNTER > 0:
                        fout.write('--------\n')
                    fout.writelines(block)
        else:
            # PATTERN
            assert first_line.startswith(':')
            end_of_name_idx = first_line.index(':',1)
            pattern_name = first_line[1:end_of_name_idx]
            pattern_name = pattern_name.replace('/','_')
            comment = ' '.join(l.strip() for l in comment_lines)
            pattern = '\n'.join(l.strip() for l in pattern_lines)
            on_counter = count_on_cells(pattern)
            pattern_name = str(on_counter).zfill(4) + '_' + pattern_name

            if filesystem_write:
                pattern_filepath = os.path.join(LEX_ASC_PATTERNS_DIR, f'{pattern_name}.txt')
                with open(pattern_filepath, 'w') as fout:
                    fout.write(comment)
                    fout.write('\n')
                    fout.write(pattern)
                    fout.write('\n')

            PATTERN_COUNTER += 1
            ptattern_counter_zfill = str(PATTERN_COUNTER).zfill(4)
            if verbose:
                print(f'{ptattern_counter_zfill}: {pattern_name}')

    return pattern_name, pattern

def clean_pattern_filesystem():
    # init (clean) LEX_ASC_PATTERNS_REF file
    for file_to_clean in [LEX_ASC_PATTERNS_REF, LEX_ASC_NON_PATTERNS]:
        with open(file_to_clean, 'w') as fout:
            fout.write('')

    # delete old patterns
    remove_pattern_files()


def process_lexicon(filesystem_write=False, verbose=True):

    all_patterns = dict()

    with open(LEX_ASC_PATH) as fin:
        lines = fin.readlines()

    num_blocks = 0
    block = []
    block_started = False
    for l in lines:
        if l.startswith(':'):
            block_started = True
        if block_started:
            if l == '\n':
                block_started = False
                num_blocks += 1
                pattern_name, pattern = process_pattern(
                    block,
                    filesystem_write = filesystem_write,
                    verbose = verbose
                )
                if pattern_name is not None:
                    all_patterns[pattern_name] = pattern
                block = []
            else:
                block.append(l)
    if verbose:
        print('------')
        print(f'Found {num_blocks} blocks')
        print(f'Found {REF_COUNTER} ref')
        print(f'Found {PATTERN_COUNTER} patterns')
        print(f'Found {NON_PATTERN_COUNTER} NON-patterns (other blocks)')
    else:
        print(f'Read {PATTERN_COUNTER} patterns from {LEX_ASC_PATH}')
    assert num_blocks == REF_COUNTER + PATTERN_COUNTER + NON_PATTERN_COUNTER

    return all_patterns

def get_lex_patterns():
    return process_lexicon(
        filesystem_write=False,
        verbose = False
    )

if __name__ == "__main__":
    clean_pattern_filesystem()
    process_lexicon(
        filesystem_write=True,
        verbose=True
    )