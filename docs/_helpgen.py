#!/usr/bin/env python

"""Generate parts of documentation from source code."""

import os
import re
import runpy
import sys
import textwrap

SOURCE = os.path.normpath(os.path.join(
    __file__,
    '..',
    '..',
    'src',
    'pdfslash',
    'slash.py',
))

OUTDIR = os.path.normpath(os.path.join(
    __file__,
    '..',
    '_help',
))

DATA = {}


def part(text, start='', start_next='', end='', end_before=''):
    new = []
    state = 'skip'  # 'skip', 'include', 'break'
    for line in text.split('\n'):
        if state == 'skip':
            if start and line.strip().startswith(start):
                new.append(line)
                state = 'include'
            elif start_next and line.strip().startswith(start_next):
                state = 'include'
        elif state == 'include':
            if end and line.strip().startswith(end):
                new.append(line)
                state = 'break'
            elif end_before and line.strip().startswith(end_before):
                state = 'break'
            else:
                new.append(line)
        else:
            break
    ret = '\n'.join(new)
    return textwrap.dedent(ret)


def get_conf(source):
    text = part(source, start_next='_CONF = {', end_before='# not used')
    new = ['[main]', '']
    for line in text.split('\n'):
        if not line.strip():
            new.append(line)
        elif line.strip().startswith('#'):
            new.append(line)
        else:
            newline = re.sub(r': \((.+?), [^)]+\),$', r' = \1', line)  # func
            newline = re.sub(r"'(.+?)'", r'\1', newline)  # quote
            newline = re.sub(r'\((.+?)\)', r'\1', newline)  # parenthesis
            new.append(newline)
    text = '\n'.join(new)
    DATA['conf'] = text


def get_nstr(mod):
    text = mod['NumParser'].__doc__
    text = part(text, start_next='Spec:', end_before='"""')
    DATA['nstr'] = text.lstrip()


def get_box(mod):
    text = mod['BoxParser'].__doc__
    text = part(text, start_next='Spec:', end_before='"""')
    DATA['box'] = text.lstrip()


def get_cmds(mod):
    c = mod['PDFSlashCmd']
    new = []
    append = new.append

    # for m in sorted(c.__dict__):
    for m in c.__dict__:
        if m.startswith('do_'):
            name = m[3:]
            doc = c.__dict__[m].__doc__
            doc = doc.strip('\n')
            doc = textwrap.dedent(doc)

            append('**%s**' % name)
            append('')
            append(doc)
            append('')

    text = '\n'.join(new)
    DATA['cmds'] = text


def get_gui(mod):
    text = mod['_tk_help']
    text = part(text, start_next='preview help:', end_before='-------')
    DATA['gui'] = text


# from tosixinch/tests/dev/argparse2rst.py
def get_commandline(mod):
    text = []
    append = text.append
    parser = mod['_build_argument_parser']()
    formatter = parser._get_formatter()
    for a in parser._actions:
        h = a.help.replace('\n', ' ')
        if h == '==SUPPRESS==':
            continue
        # o = ', '.join(a.option_strings)
        o = formatter._format_action_invocation(a)
        # o = _check_too_long_choices(o)
        append('.. option:: %s\n\n    %s\n' % (o, h))
        d = a.default
        c = a.choices
        if d and d != '==SUPPRESS==':
            append('        default=%s\n' % d)
        if c:
            append('        choices=%s\n' % ', '.join(c))

    text = '\n'.join(text)
    DATA['commandline'] = text


def build(mod, source):
    get_conf(source)
    get_nstr(mod)
    get_box(mod)
    get_cmds(mod)
    get_gui(mod)
    get_commandline(mod)


# not used
def check():
    getmtime = os.path.getmtime
    boxfile = os.path.join(OUTDIR, 'box.txt')
    return getmtime(SOURCE) > getmtime(boxfile)


# not used
def rst_format():
    text = []
    for k, v in DATA.items():
        t = '.. |%s| replace:: %s\n\n\n' % (k, v)
        text.append(t)
    text = ''.join(text)
    text = '"""%s"""' % text
    return text


def write():
    for k, v in DATA.items():
        fname = os.path.join(OUTDIR, k + '.txt')
        with open(fname, 'w') as f:
            f.write(v)


def delete_files():
    with os.scandir(OUTDIR) as it:
        for entry in it:
            if entry.name.endswith('.txt') and entry.is_file():
                os.remove(entry.path)

def main():
    # if not check():
    #     print('no chnage -- _helpgen.py')
    #     return

    mod = runpy.run_path(SOURCE)
    with open(SOURCE) as f:
        source = f.read()

    build(mod, source)
    write()


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '-d':
        delete_files()
    else:
        main()
