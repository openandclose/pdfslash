#!/usr/bin/env python

"""Crop PDF margins from interactive interpreter."""

# Conventions (As long as reasonable):
#
# Use 'index', 'ind' or 'i' for 0-based sequence,
# and 'number', 'num' or 'n' for 1-based.
#
# Use 'img' for numpy pixel data,
# and 'image' for actual image (with header etc.).
#
# 'MediaBox' and 'CropBox':
# raw PDF values
# pre-rotation. (left, bottom, right, top). y-ascendant. real number strings.
#
# 'mbox' and 'cbox':
# PyMuPDF values of Page.mediabox and Page.cropbox
# pre-rotation. (left, top, right, bottom). y-descendant. Python floats.
# While mbox values are superficially the same as PDF values,
# cbox values are in the coords in which mbox's top (y0) is moved to 0.
# see https://pymupdf.readthedocs.io/en/latest/glossary.html#MediaBox
# (This is necessary, in general, when you transform y-axis direction).
#
# 'mediabox' and 'cropbox':
# The values the program uses.
# after-rotation. (left, top, right, bottom). y-descendant. integers.
# After y-axis transform, mediabox's left-top (x0, y0) is moved to (0, 0),
# both for mediabox and cropbox (and then rotation is applied).
# (This is necessary, in general, when you handle image or pixel rectangles,
# like in GUI, but for consistency, they are used in all other interfaces).
#
# Example:
# When Mediabox and CropBox are both '[1 2 100.1 200.2]',
# mbox is (1.0, 2.0, 100.0999984741211, 200.1999969482422),
# cbox is (1.0, 0.0, 100.0999984741211, 198.1999969482422),
# mediabox is (0, 0, 99, 198),
# cropbox is (0, 0, 99, 198).
#
# 'box'
# A new box (cropbox candidate) created in the program, is just called 'box'.
#
# Trying to use 'box' for one box in a page, 'boxes' for boxes in a page,
# 'pageboxes' for list of boxes in a collection of pages.


import argparse
import code
import configparser
import cmd
import math
import os
import re
import subprocess
import shlex
import sys
import time
import traceback
import tkinter as tk
import zlib

from collections.abc import MutableSequence

try:
    import readline
except ImportError:
    readline = None

try:
    import numpy
except ImportError:
    numpy = None

if numpy:
    import numpy.lib.stride_tricks

    numpy.seterr(all='raise')
    UINT8 = numpy.uint8
    INT = numpy.int_

try:
    import pymupdf as fitz  # since v1.24.3 (2024/05/10)
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

if fitz:
    import fitz.utils


# Global NumParser instance
# While user can technically customize NumParser for interpreter input,
# other parts of the code must use a normal version of NumParser instance.
g_numparser = None


_ENV_VAR_DIR = 'PDFSLASH_DIR'
_CONFIG_FILENAME = 'pdfslash.ini'

_CONFIGFUNC = {
    'str': str,
    'int': int,
    'float': float,
    'two_floats': lambda s: tuple(map(float, s.split(',', maxsplit=1))),
}

_CONF = {
    # The ratio of computer display pixel, to PDF pixel.
    'device_pixel_ratio': (1.0, 'float'),

    # Gui window position to the display margin (x and y).
    # 0.0, 0.0: top-left aligned
    # 0.5, 0.5: center
    # 1.0, 1.0: bottom-right aligned
    'winpos': ((0.5, 0.5), 'two_floats'),

    # Max pages to sample, to create a merge image (one group) in GUI.
    # So when running 'preview 1-600',
    # the program is acutually showing
    # only this number of arbitrarily selected pages.
    # '15' is briss' default.
    'max_merge_pages': (15, 'int'),

    # Merge method to use in page-image-merging,
    # either 'briss' (default) or 'simple'.
    # 'simple' is a bit faster,
    # and *may* work in some cases where 'briss' doesn't.
    'merge': ('briss', 'str'),
}


COLORS = {
    # https://en.wikipedia.org/wiki/Web_colors#Extended_colors
    'blue': '#0000ff',  # Blue
    'lightblue': '#8080ff',
    'green': '#006400',  # DarkGreen
    'orange': '#FF4500',  # OrangeRed
    'red': '#FF0000',  # Red
}


# for tests
_PRINT_TIME = False
_TIMES = []

_SAVE_IMG = False


def _time(msg=''):
    # Always use it in pairs (start and end), nesting is possible.
    if _PRINT_TIME:
        t = time.time()
        if msg in ('', 'start'):
            _TIMES.append(t)
        elif msg in ('pop',):
            t0 = _TIMES.pop()
        else:
            t0 = _TIMES.pop()
            print('    [time] %-32s: %.4fs' % (msg, (t - t0)))


def pdfbox2box(box):
    # Convert formatted PDF box string to Python float tuple.
    box = box.strip()
    box = box[1:-1]
    box = box.split()
    box = [float(v) for v in box]
    return box


def num2ind(numbers):  # numbers to indices (1-based to 0-based)
    return tuple(n - 1 for n in numbers)


def ind2num(indices):  # indices to numbers (0-based to 1-based)
    return tuple(n + 1 for n in indices)


def getsize(box):
    left, top, right, bottom = box
    return int(right - left), int(bottom - top)


def shift_box(box, pos):
    return tuple(b + p for b, p in zip(box, pos))


def groupby(seq, key=None):
    """Iterate on grouped items collected from unsorted Sequence.

    https://code.activestate.com/recipes/580800-groupby-for-unsorted-input/#c1
    """
    if key is None:
        key = lambda x: x
    _groups = {}
    for i, item in enumerate(seq):
        try:
            _groups[key(item)].append(i)
        except KeyError:
            _groups[key(item)] = [i]
    for k, group in _groups.items():
        yield k, tuple(seq[i] for i in group)


def filter_numbers(numbers, which=0, need_indices=False):
    filters = {
        0: lambda x: True,  # all
        1: lambda x: x % 2,  # odds
        2: lambda x: not x % 2,  # evens
    }
    func = filters[which]
    nums = tuple(n for n in numbers if func(n))
    if not need_indices:
        return nums
    number_indices = [i for i, n in enumerate(numbers) if func(n)]
    return nums, number_indices  # note: tuple and list


def rotate(w, h, rot, box):
    # Rotate box in PDF mediabox coordinates (0, 0, w, h).
    # rot must be a multiple of 90.
    rot = rot % 360
    if rot == 0:
        new = box
    elif rot == 90:
        new = [h - box[3], box[0], h - box[1], box[2]]
    elif rot == 180:
        new = [w - box[2], h - box[3], w - box[0], h - box[1]]
    elif rot == 270:
        new = [box[1], w - box[2], box[3], w - box[0]]
    else:
        new = box  # illegal in PDF reference
    return tuple(new)


def unrotate(w, h, rot, box):
    if rot in (90, 270):
        w, h = h, w
    return rotate(w, h, 360 - rot, box)


def get_checksum(fname):
    fpath = os.path.abspath(fname)
    with open(fpath, 'rb') as f:
        return '%08x' % (zlib.crc32(f.read()) & 0xffffffff)


class PDFSlashError(Exception):
    """Errors the program defines."""

    msg = ''

    def __init__(self, *args):
        args = args or ()
        message = self.msg % args
        super().__init__(message)

    def tostring(self):
        return '%s: %s' % (self.__class__.__name__, self.args[0])


class UserInputError(PDFSlashError):
    """Errors on user interface, which the program should supress."""


class DuplicateBoxError(UserInputError):
    """Raise when adding the same box already there in a page."""

    msg = 'cannot add the duplicate box; page: %d, box: %s'


class NoBoxToProcessError(UserInputError):
    """Raise when trying to edit non-existent box in a page."""

    msg = 'cannot process non-existent box; page: %d'


class UndoError(UserInputError):
    """Raise when not undoable."""

    msg = 'cannot undo (reached the first).'


class RedoError(UserInputError):
    """Raise when not redoable."""

    msg = 'cannot redo (reached the last)'


class _Stack(object):
    """Implement a stack for single branch undo and redo."""

    def __init__(self):
        self._stack = []
        self.pos = -1

    def push(self, data):
        self._stack[self.pos + 1:] = [data]
        self.pos += 1

    def undo(self):
        if not self.undoable:
            raise UndoError
        data = self._stack[self.pos]
        self.pos -= 1
        return data

    def redo(self):
        if not self.redoable:
            raise RedoError
        data = self._stack[self.pos + 1]
        self.pos += 1
        return data

    @property
    def undoable(self):
        return self.pos > -1

    @property
    def redoable(self):
        return self.pos < len(self._stack) - 1


class _StackContext(object):
    """Define command set (one unit of changes for undo and redo).

    Use it as decorator to class method.
    The class should define 'initialize', 'push' and 'rollback'.
    """

    def __enter__(self):
        self.handler.initialize()

    def __exit__(self, exc_type, exc, exc_tb):
        if exc_type is None:
            self.handler.push()
        else:
            self.handler.rollback()
            return self.handle_err(exc_type, exc, exc_tb)

    def __call__(self, method):
        self.__wrapped__ = method

        def inner(handler, *args, **kwargs):
            self.handler = handler  # handler is method's class (another self).
            with self:
                # return method(*args, **kwargs)
                return method(handler, *args, **kwargs)
        return inner

    def handle_err(self, exc_type, exc, exc_tb):
        if issubclass(exc_type, UserInputError):
            print(exc.tostring())
            return True
        else:
            return False


stackcontext = _StackContext()


class _Stacker(object):
    """Process _Stack.

    Operations (op) are 'add', 'replace' and 'remove',
    imitating (a subset of) JSON Patch (RFC 6902).
    """

    def __init__(self, data):
        self._data = data
        self._stack = _Stack()
        self._commands = []
        self._msg = ''

    def initialize(self):
        self._commands = []
        self._msg = ''

    def push(self):
        # push even blank commands
        self._stack.push((self._commands, self._msg))

    def rollback(self):
        self._rollback(self._commands)
        self._msg = None

    def _get_item(self, obj, key):
        if hasattr(obj, '__getitem__'):
            return obj[key]
        else:
            return getattr(obj, key)

    def _set_item(self, obj, key, value):
        if hasattr(obj, '__setitem__'):
            obj[key] = value
        else:
            setattr(obj, key, value)

    def _add_item(self, obj, key, value):
        if hasattr(obj, 'insert'):
            obj.insert(key, value)
        elif hasattr(obj, '__setitem__'):
            obj[key] = value
        else:
            setattr(obj, key, value)

    def _get(self, keys=''):
        obj = self._data
        if keys == '':
            return obj
        for key in keys:
            obj = self._get_item(obj, key)
        return obj

    @stackcontext
    def set(self, commands, msg=None):
        self._commands = []
        for command in commands:
            for command in self.preprocess(command):
                ret, old_val = self._set(command)
                if ret == 0:
                    self._commands.append(command + (old_val,))

        if msg is not None:
            self._msg = msg

    def preprocess(self, command):
        op, *args = command
        if op in ('add', 'replace', 'remove'):
            yield command
        else:
            yield from op(*args)

    def _set(self, command):
        op, keys, value = command
        obj = self._get(keys[:-1])
        if op == 'add':
            old_val = None
        else:
            old_val = self._get_item(obj, keys[-1])
        if op == 'replace' and value == old_val:
            return 1, None  # skip
        self._apply(op, obj, keys[-1], value)
        return 0, old_val

    def _apply(self, op, obj, key, value):
        if op == 'add':
            self._add_item(obj, key, value)
        elif op == 'replace':
            self._set_item(obj, key, value)
        elif op == 'remove':
            del obj[key]

    def execute(self, command):
        op, keys, value, *old_val = command
        obj = self._get(keys[:-1])
        self._apply(op, obj, keys[-1], value)

    def _reverse_command(self, command):
        op, keys, value, old_val = command
        if op == 'add':
            return 'remove', keys, None
        elif op == 'replace':
            return 'replace', keys, old_val
        elif op == 'remove':
            return 'add', keys, old_val

    def _rollback(self, commands):
        commands = (self._reverse_command(c) for c in reversed(commands))
        for command in commands:
            self.execute(command)

    def undo(self):
        ret = self._stack.undo()
        commands, msg = ret
        self._rollback(commands)
        return msg

    def redo(self):
        ret = self._stack.redo()
        commands, msg = ret
        for command in commands:
            self.execute(command)
        return msg

    def export(self):
        stack = self._stack._stack
        pos = self._stack.pos
        return [msg for commands, msg in stack[:pos + 1]]


class _Boxes(MutableSequence):
    """Behave as a box list, auto-create box dict (item-keyed dict)."""

    def __init__(self, num, boxdict, initlist=None):
        self._num = num
        self._boxdict = boxdict
        self.data = initlist or []

    def __getitem__(self, index):
        try:
            return self.data[index]
        except IndexError:
            raise NoBoxToProcessError(self._num)

    def __setitem__(self, index, item):
        self.check_duplicate(item)
        old_item = self[index]
        self.data[index] = item
        self._boxdict.replace(self._num, item, old_item)

    def __delitem__(self, index):
        item = self[index]
        del self.data[index]
        self._boxdict.remove(self._num, item)

    def __len__(self):
        return len(self.data)

    def insert(self, index, item):
        self.check_duplicate(item)
        self.data.insert(index, item)
        self._boxdict.add(self._num, item)

    def check_duplicate(self, item):
        if item in self.data:
            raise DuplicateBoxError(self._num, str(item))

    def __repr__(self):
        return repr(self.data)

    def __iter__(self):
        return self.data.__iter__()


class _BoxData(object):
    """Manage a collection of _Boxes (future cropboxes).

    Define additional operations (op):
    'append', 'overwrite', 'modify', 'discard', 'clear' and 'set_each'.
    """

    def __init__(self, numbers):
        self.numbers = numbers
        self.boxdict = _Boxdict(self)
        self.boxes = [_Boxes(n, self.boxdict) for n in self.numbers]
        self.stacker = _Stacker(self.boxes)

    def set(self, method, numbers, box=None, old_box=None, msg=None):
        command_set = []
        for n in numbers:
            command = method, n - 1, box, old_box
            command_set.append(command)

        self.stacker.set(command_set, msg)

    def set_each(self, commands, msg=None):
        # Note: multiple edits in a same page don't work in most cases.
        command_set = []
        for c in commands:
            opname, n, *boxes = c
            op = getattr(self, '_' + opname)
            box, *old_box = boxes
            old_box = old_box[0] if old_box else None
            command = op, n - 1, box, old_box
            command_set.append(command)

        self.stacker.set(command_set, msg)

    def append(self, numbers, box, msg=None):
        self.set(self._append, numbers, box, msg=msg)

    def overwrite(self, numbers, box, msg=None):
        self.set(self._overwrite, numbers, box, msg=msg)

    def modify(self, numbers, box, old_box, msg=None):
        self.set(self._modify, numbers, box, old_box, msg=msg)

    def discard(self, numbers, box, msg=None):
        self.set(self._discard, numbers, None, box, msg=msg)

    def clear(self, numbers, msg=None):
        self.set(self._clear, numbers, msg=msg)

    def _append(self, i, box, old_box):
        index = len(self.boxes[i])
        return [('add', (i, index), tuple(box))]

    def _overwrite(self, i, box, old_box):
        commands = self._clear(i, box, old_box)
        commands2 = [('add', (i, 0), tuple(box))]
        return commands + commands2

    def _modify(self, i, box, old_box):
        boxes = self.boxes[i]
        index = boxes.index(old_box)
        commands = [('replace', (i, index), tuple(box))]
        return commands

    def _discard(self, i, box, old_box):
        boxes = self.boxes[i]
        index = boxes.index(old_box)
        commands = [('remove', (i, index), None)]
        return commands

    def _clear(self, i, box, old_box):
        commands = []
        boxes = self.boxes[i]
        for index in reversed(range(len(boxes))):
            command = 'remove', (i, index), None
            commands.append(command)
        return commands

    def undo(self):
        return self.stacker.undo()

    def redo(self):
        return self.stacker.redo()

    def __repr__(self):
        return '\n'.join(repr(b) for b in self.boxes)


class _Boxdict(object):
    """Manage box-keyed version of box data."""

    def __init__(self, boxdata):
        self.boxdata = boxdata
        self.rects = {}

    def add(self, number, box):
        if box not in self.rects:
            self.rects[box] = []
        numbers = self.rects[box]
        if not numbers:
            numbers.append(number)
        else:
            for i, n in enumerate(numbers):
                if number < n:
                    numbers.insert(i, number)
                    return
            numbers.append(number)

    def replace(self, number, box, old_box):
        self.remove(number, old_box)
        self.add(number, box)

    def remove(self, number, box):
        numbers = self.rects[box]
        numbers.remove(number)


class _Page(object):
    """Define page data attributes."""

    def __init__(self, pages, number):
        self.pages = pages
        self.number = number

    @property
    def mediabox(self):
        return self.pages.mediaboxes[self.number - 1]

    @property
    def cropbox(self):
        return self.pages.cropboxes[self.number - 1]

    @property
    def boxes(self):
        return self.pages.boxdata.boxes[self.number - 1]

    @property
    def selected(self):
        return self.pages.selected[self.number - 1]

    @property
    def fixed(self):
        return self.pages.fixed[self.number - 1]

    def __len__(self):
        return len(self.boxes)

    def __iter__(self):
        return self.boxes.__iter__()

    def tostring(self):
        selected = 's' if self.selected else ' '
        fixed = 'f' if self.fixed else ' '
        box = '%d,%d,%d,%d' % self.cropbox
        fmt = '%s%s %4d  source: %s'
        if not self.boxes:
            return fmt % (selected, fixed, self.number, box)
        else:
            fmt += '  now: %s'
            newbox = ['%d,%d,%d,%d' % box for box in self.boxes]
            newbox = '; '.join(newbox)
            return fmt % (selected, fixed, self.number, box, newbox)


class _Pages(object):
    """Define page data interface."""

    def __init__(self, mediaboxes, cropboxes):
        self.mediaboxes = mediaboxes
        self.cropboxes = cropboxes
        self.numbers = tuple(range(1, len(mediaboxes) + 1))
        self.boxdata = _BoxData(self.numbers)
        self.pages = [_Page(self, n) for n in self.numbers]

        self.selected = [1 for _ in range(len(self.numbers))]
        self.fixed = [0 for _ in range(len(self.numbers))]

        global g_numparser
        g_numparser = NumParser(len(mediaboxes))

    def __len__(self):
        return len(self.pages)

    def __iter__(self):
        return self.pages.__iter__()

    def __getitem__(self, number):
        if number < 1:
            raise IndexError('numbers are 1-based. got %r' % number)
        return self.pages[number - 1]

    def selectable(self, numbers):
        return [n for n in numbers if self.selected[n - 1]]

    def modifiable(self, numbers):
        return [n for n in numbers
            if self.selected[n - 1] and not self.fixed[n - 1]]

    def select(self, numbers):
        for n in numbers:
            self.selected[n - 1] = 1

    def unselect(self, numbers):
        for n in numbers:
            self.selected[n - 1] = 0

    def fix(self, numbers):
        for n in numbers:
            self.fixed[n - 1] = 1

    def unfix(self, numbers):
        for n in numbers:
            self.fixed[n - 1] = 0

    def format_msg(self, op, numbers, box='', new_box=''):
        if isinstance(numbers, int):
            numbers = (numbers,)
        nstr = g_numparser.unparse(numbers)
        if box:
            box = '%d,%d,%d,%d' % box
        if new_box:
            new_box = '%d,%d,%d,%d' % new_box
        ret = [s for s in (op, nstr, box, new_box) if s]
        return ' '.join(ret).strip()

    def append(self, numbers, box, msg=None):
        numbers = self.modifiable(numbers)
        self.verify(numbers, box)
        msg = msg or self.format_msg('append', numbers, box)
        self.boxdata.append(numbers, box, msg=msg)

    def overwrite(self, numbers, box, msg=None):
        numbers = self.modifiable(numbers)
        self.verify(numbers, box)
        msg = msg or self.format_msg('overwrite', numbers, box)
        self.boxdata.overwrite(numbers, box, msg=msg)

    def set_each(self, commands_, msg=None):
        commands = []
        for command in commands_:
            op, n, *boxes = command
            numbers = (n,)
            if self.modifiable(numbers):
                self.verify(numbers, boxes[-1])
                commands.append(command)

        if msg is None:
            start = '# start set_each commands.'
            end = '# end set_each commands.'
            msgs = [start]
            for command in commands:
                msgs.append(self.format_msg(*command))
            msgs.append(end)
            msg = '\n'.join(msgs)

        self.boxdata.set_each(commands, msg=msg)

    def modify(self, numbers, old_box, new_box, msg=None):
        numbers = self.modifiable(numbers)
        self.verify(numbers, new_box)
        msg = msg or self.format_msg('modify', numbers, old_box, new_box)
        self.boxdata.modify(numbers, new_box, old_box, msg=msg)

    def discard(self, numbers, box, msg=None):
        numbers = self.modifiable(numbers)
        msg = msg or self.format_msg('discard', numbers, box)
        self.boxdata.discard(numbers, box, msg=msg)

    def clear(self, numbers, msg=None):
        numbers = self.modifiable(numbers)
        msg = msg or self.format_msg('clear', numbers)
        self.boxdata.clear(numbers, msg=msg)

    def check_msg(self):
        return self.boxdata.stacker._msg

    def _format_undo_msg(self, msg, which='undo'):
        if not msg:
            return
        if '\n' in msg:
            start = ['# %s:' % which]
            msg = ['# %s' % m for m in msg.split('\n')]
            return '\n'.join(start + msg)
        else:
            return '# %s - %s' % (which, msg)

    def undo(self):
        msg = self.boxdata.undo()
        return self._format_undo_msg(msg, 'undo')

    def redo(self):
        msg = self.boxdata.redo()
        return self._format_undo_msg(msg, 'redo')

    def verify(self, numbers, box=None):
        if not numbers:
            raise ValueError('No page numbers')
        if box:
            self._verify_crop(numbers, box)

    def _verify_crop(self, numbers, box):
        self._verify_box(numbers, box)

    def _verify_box(self, numbers, box):
        right = min(self[n].mediabox[2] for n in numbers)
        bottom = min(self[n].mediabox[3] for n in numbers)
        min_box = 0, 0, right, bottom

        fmt = 'box is not inside source mediabox. box: %d,%d,%d,%d.'
        if box[0] < min_box[0] or box[1] < min_box[1]:
            raise ValueError(fmt % box)
        if min_box[2] < box[2] or min_box[3] < box[3]:
            raise ValueError(fmt % box)

    def get_boxes(self, number, fallback=True):
        page = self[number]
        if fallback:
            return page.boxes.data or [page.cropbox]
        else:
            return page.boxes.data

    def get_pageboxes(self, numbers=None, fallback=True):
        numbers = numbers or self.numbers
        return [self.get_boxes(n, fallback=fallback) for n in numbers]

    def get_boxes_flattened(self, numbers):
        # Used in pdf backend write (Document.write).
        # e.g. when A, B, C are boxes,
        # [5, [A, B, C]] -> [5, 5, 5], [A, B, C]
        new_numbers = []
        new_boxes = []
        for n in numbers:
            boxes = self.get_boxes(n, fallback=False)
            if boxes == []:
                new_numbers.append(n)
                new_boxes.append(None)
                continue
            for box in boxes:
                new_numbers.append(n)
                new_boxes.append(box)
        return new_numbers, new_boxes

    def is_single_boxes(self, numbers):  # Each page has zero or one box.
        return all(len(self.get_boxes(n)) == 1 for n in numbers)

    def tostring(self, numbers=None):
        numbers = numbers or self.numbers
        return '\n'.join([self[n].tostring() for n in numbers])


class _ImgProxy(object):
    """Behave as numpy array, but load the data only when necessary.

    It takes time for backend to build raster data from PDF,
    while the program may only need limited pages of data.

    Keep blank array with the same shape as the actual data,
    handle only first axis indexing (page filtering),
    returning a new instance.

    If explicitly requested (.load), return the actual data.
    """

    _errorfmt = ('This is numpy array proxy. '
        'It only accepts first axis indexing (int or list). Got: %r.')

    def __init__(self, array, loader, root, current=None, _loaded=None):
        self.array = array
        self.loader = loader
        if not isinstance(root, numpy.ndarray):
            root = numpy.asarray(root)
        self.root = root  # root (global) indices
        if current is None:
            current = numpy.arange(len(root))
        self.current = current  # current (in-group) indices
        self._loaded = _loaded or []
        self._zero_image = None

    def __getitem__(self, keys):
        if isinstance(keys, int):
            keys = [keys]
        try:
            current = self.current[keys]
        except IndexError:
            raise IndexError(self._errorfmt % keys)
        array, loader, root = self.array, self.loader, self.root
        return self.__class__(array, loader, root, current, self._loaded)

    def load(self):
        cnt = 0
        indices = self.root[self.current]
        for i, index in zip(self.current, indices):
            if index not in self._loaded:
                index = int(index)  # from numpy int to python int
                self.array[i] = self.loader(index)
                self._loaded.append(index)
                cnt += 1

        array = self.array[self.current]
        return array, cnt

    def load_zeros(self):
        if self._zero_image is None:
            shape = self.array.shape[1:]
            self._zero_image = numpy.zeros(shape, dtype=UINT8)
        return self._zero_image

    def __len__(self):
        return len(self.current)


class _ImgGroup(object):
    """Group imgs by sizes."""

    def __init__(self, doc, mediaboxes, cropboxes):
        self._doc = doc
        self._msizes = [self._get_size(box) for box in mediaboxes]
        self._csizes = [self._get_size(box) for box in cropboxes]
        self._indices = tuple(range(len(mediaboxes)))
        self._load_groups()

    def _get_size(self, box):
        col, row = getsize(box)
        return row, col

    def _groupby(self, indices):
        key = lambda x: self._msizes[x]
        yield from groupby(indices, key=key)

    def _load_groups(self):
        """Create numpy array image data (imgs) and put them into groups.

        ``groups`` are a dict in which key is size (tuple of row and column),
        and value is a list of imgs.

        ``_table`` is a dict in which key is img index (page_number - 1),
        value is a tuple of size and in-group index.
        """
        table, groups = {}, {}
        for size, indices in self._groupby(self._indices):
            shape = len(indices), *size
            array = numpy.zeros(shape, dtype=UINT8)
            for i, index in enumerate(indices):
                table[index] = (size, i)
            array = _ImgProxy(array, self._get_img, indices)
            groups[size] = array
        self._table = table
        self._groups = groups

    def _get_img(self, index):
        return self._doc.backend.get_img(index + 1)

    def get(self, indices=None, kind='subgroup'):
        indices = indices or self._indices
        if kind == 'group':
            yield from self.get_groups(indices)
        elif kind == 'subgroup':
            yield from self.get_subgroups(indices)
        elif kind == 'single':
            yield from self.get_singles(indices)
        else:
            fmt = "kind is one of 'group', 'subgroup' or 'single'. got: %r"
            raise ValueError(fmt % kind)

    def _build_metadata(self, indices, g_num, sub1_num=None, sub2_num=None):
        return {
            'root_indices': indices,  # original input (from ui)
            'g_num': g_num,  # number of groups (including optinal subgroups)
            'sub1_num': sub1_num,  # number of mediabox groups
            'sub2_num': sub2_num,  # number of cropbox groups (in a sub1 group)
        }

    def get_groups(self, indices):
        """Yield group (same-size pages) selected from indices."""
        g_num = self._get_number_of_groups(indices)
        meta = self._build_metadata(indices, g_num)

        for size, indices in self._groupby(indices):
            array = self._groups[size]
            g_indices = [self._table[index][1] for index in indices]
            yield meta, indices, array[g_indices]

    def _get_number_of_groups(self, indices):
        """Pre-calculate the number of groups."""
        sizes = [self._msizes[index] for index in indices]
        return len(set(sizes))

    def get_subgroups(self, indices):
        """Yield subgroup (same cropboxes) selected from indices."""
        g_num = self._get_number_of_all_subgroups(indices)
        for meta, indices, array in self.get_groups(indices):
            meta['sub1_num'] = meta['g_num']
            meta['g_num'] = g_num
            meta['sub2_num'] = self._get_number_of_subgroups(indices)
            for box, s_indices in self._subgroupby(indices):
                subg_indices = [indices.index(index) for index in s_indices]
                yield meta, s_indices, array[subg_indices]

    def _subgroupby(self, indices):
        key = lambda x: self._csizes[x]
        yield from groupby(indices, key=key)

    def _get_number_of_all_subgroups(self, indices):
        nums = 0
        for size in set([self._msizes[index] for index in indices]):
            csizes = [self._csizes[index] for index in indices
                if self._msizes[index] == size]
            nums += len(set(csizes))
        return nums

    def _get_number_of_subgroups(self, indices):
        """Pre-calculate the number of goups and subgroups."""
        csizes = [self._csizes[index] for index in indices]
        return len(set(csizes))

    def get_singles(self, indices):
        """Yield imgs one by one, in indices order."""
        meta = self._build_metadata(indices, len(indices))
        for index in indices:
            yield meta, [index], self[index]

    def __getitem__(self, key):
        size, g_index = self._table[key]
        return self._groups[size][g_index]


class _ImgSet(object):
    """Merge imgs and create derivatives (both, odds, evens)."""

    def __init__(self, doc, imgs, kind='subgroup'):
        self._doc = doc
        self.imgs = imgs
        self.kind = kind

    def get(self, indices):
        for meta, indices, imgs in self.imgs.get(indices, self.kind):
            both = (indices, self._get_img(imgs, indices, save=True))

            odds, o_indices = self._get_odds(indices)
            odds = (odds, self._get_img(imgs[o_indices], odds))

            evens, e_indices = self._get_evens(indices)
            evens = (evens, self._get_img(imgs[e_indices], evens))

            yield meta, both, odds, evens

    def _get_odds(self, indices):
        # odd page 'numbers' are even img 'indices'
        return filter_numbers(indices, 2, need_indices=True)

    def _get_evens(self, indices):
        return filter_numbers(indices, 1, need_indices=True)

    def _get_img(self, imgs, indices, save=False):
        imgs = self._select_imgs(imgs)
        if len(imgs) == 0:
            # Sometimes there are no odd or even pages,
            # then gui draws a black image.
            return imgs.load_zeros()
        else:
            _time('start')
            imgs, cnt = imgs.load()
            if cnt == 0:
                _time('pop')
            else:
                _time('PDF to image, %d pages' % cnt)
            name = self._doc.conf['merge']
            _time('start')
            img = self._doc.imgmerger.merge(imgs, indices, method_name=name)
            _time('merge image, %d pages' % len(imgs))
            if _SAVE_IMG and save and fitz:
                self._save_img(img, save)
            return img

    def _select_imgs(self, imgs):
        max_ = self._doc.conf['max_merge_pages']
        length = len(imgs)
        if length <= max_:
            return imgs
        indices = numpy.linspace(0, length - 1, num=max_, dtype=INT)
        return imgs[indices]

    def _save_img(self, img, save=False):
        samples = img.tobytes()
        h, w = img.shape
        pixmap = fitz.Pixmap(fitz.csGRAY, w, h, samples, 0)
        t = time.time()
        print('    saving image...')
        pixmap.save('merged_image_%d.png' % t)


class _ImageData(object):
    """Manage image data and expose current states (for tkinter)."""

    def __init__(self, doc, indices, kind='subgroup'):
        self._doc = doc
        self._indices = indices  # original indices

        self._imgset = _ImgSet(doc, doc.imgs, kind)
        self._it = self._imgset.get(indices)

        dev_scale = doc.conf['device_pixel_ratio']
        self._scaling = _Scaling(dev_scale=dev_scale)

        # persistent img cache (in the Document class)
        cache = doc._img_cache
        if not cache.get(indices):
            cache[indices] = {
                'metadata': {},
                'data': {},
            }
        self._d_cache = cache[indices]['data']
        self._d_metadata = cache[indices]['metadata']

        # temporary image cache, for each gui invocation.
        self._cache = {}

        self.g_num = self._d_metadata.get('g_num')
        self.sub1_num = self._d_metadata.get('sub1_num')
        self.sub2_num = self._d_metadata.get('sub2_num')
        self.g_index = -1  # current gruop index

        self.img = None  # current img
        self.image = None  # current PhotoImage
        self.indices = None  # current imgs indices
        self.numbers = None  # current page numbers (list of integers)
        self.nstr = None  # current page numbers (short str)
        self.im_state = 0  # 0, 1 or 2 (both, odds or evens)

        self._width = None  # current img width
        self._height = None  # current img height
        self.width = None  # current scaled image width
        self.height = None  # current scaled image height

        self.rects = _Rects(self)  # current pageboxes

        self._stackdata = None  # g_index, im_state, _scaling._scale
        self._stacker = None

    def next(self):
        index = self.g_index + 1
        if index == self.g_num:
            index = 0

        if index >= len(self._cache):
            data = next(self._it)
            self._d_metadata.update(data[0])
            self.g_num = data[0]['g_num']
            self._d_cache[index] = data[1:]

            if index == self.g_num - 1:  # done with the iterator
                try:
                    next(self._it)
                except StopIteration:
                    pass

        self.update(g_index=index)

        if self._stacker is None:
            self._stackdata = self._get_stack_data()
            self._stacker = _Stacker(self._stackdata)

    def prev(self):
        index = self.g_index - 1
        if index < 0:
            if len(self._cache) < self.g_num:
                raise LookupError
            index = self.g_num - 1

        self.update(g_index=index)

    def update(self, g_index=None, im_state=None, scale=None,
            reset_rects=True):
        if reset_rects and self.rects.active_index != 0:
            self.rects.reset_active()

        if g_index is not None:
            self.g_index = g_index
        if im_state is not None:
            self.im_state = im_state
        if scale is not None:
            self._scaling._set(scale)

        if g_index is not None or im_state is not None:
            self._set_img()
        if g_index is not None or im_state is not None or scale is not None:
            self._set_image()

    def _set_img(self):
        cache = self._d_cache[self.g_index]
        state = self.im_state  # 0, 1 or 2

        indices, img = cache[state]
        self.numbers = ind2num(indices)
        self.nstr = g_numparser.unparse(self.numbers)
        self._height, self._width = img.shape
        self.indices, self.img = indices, img

    def _set_image(self):
        cache = self._cache.get(self.g_index)
        if not cache:
            cache = self._cache[self.g_index] = [{}, {}, {}]

        state = self.im_state
        scale = self._scaling.scale
        if not cache[state].get(scale):
            cache[state][scale] = self._load_image()
        self.image = cache[state][scale]
        self.width, self.height = self.image.width(), self.image.height()

    def _load_image(self):
        img = self._scaling.get_img(self.img)
        height, width = img.shape
        return self._load_image_impl(img, width, height)

    def _load_image_impl(self, img, w, h):
        pgm_header = b'P5 %d %d 255 ' % (w, h)
        data = pgm_header + img.tobytes()
        return tk.PhotoImage(data=data, format='ppm')

    def _zoom(self, which='zoom_in'):  # which: 'zoom_in' or 'zoom_out'
        if which == 'zoom_in':
            scale = self._scaling._next()
        else:
            scale = self._scaling._prev()
        if scale is None:  # no change
            return
        self.update(scale=scale, reset_rects=False)

    def _get_stack_data(self):
        return [self.g_index, self.im_state, self._scaling._scale]

    def _set_stack_data(self):
        g_index, im_state, scale = self._stackdata
        changes = [None, None, None]
        changes[0] = None if self.g_index == g_index else g_index
        changes[1] = None if self.im_state == im_state else im_state
        changes[2] = None if self._scaling._scale == scale else scale

        self.update(*changes)

    def _set(self):
        g_index, im_state, scale = self._get_stack_data()
        commands = [
            ('replace', (0,), g_index),
            ('replace', (1,), im_state),
            ('replace', (2,), scale),
        ]
        self._stacker.set(commands)

    def undo(self):
        self._stacker.undo()
        self._set_stack_data()
        return self.rects.undo()

    def redo(self):
        self._stacker.redo()
        self._set_stack_data()
        return self.rects.redo()


class ImgMergerBase(object):
    """Define imgs merging interface."""

    def __init__(self):
        self._cache = {}

    def merge(self, imgs):
        pass


class ImgMerger(ImgMergerBase):
    """Implement actual ImgMerger."""

    def merge(self, imgs, indices, **kwargs):
        if len(imgs) == 1:
            return self.singlepage(imgs)

        name = kwargs.pop('method_name')
        method = getattr(self, name, None)
        if not method:
            print('Invald merge method name: %r' % name)
            return self.zeros(imgs)

        cache = self._cache.get(name)
        if cache is None:
            self._cache[name] = {}
        cache = self._cache[name].get(indices)
        if cache is None:
            img = method(imgs)
            if not isinstance(img, numpy.ndarray):
                print('Invald merge method name: %r' % name)
                return self.zeros(imgs)
            cache = self._cache[name][indices] = img
        return cache

    # when method name is invalid, return black img.
    def zeros(self, imgs):
        shape = imgs.shape[1:]
        return numpy.zeros(shape, dtype=UINT8)

    # when dealing only one img,
    # return similar-looking pale img rather than the original.
    def singlepage(self, imgs):
        img = 255 - (255 - imgs[0]) // 3
        return img.astype(UINT8)

    def briss(self, imgs):
        """Implement briss's merger method (in ClusterImageData.java).

        https://github.com/fatso83/briss-archived
        """
        img = 255 - numpy.std(imgs, axis=0)
        return img.astype(UINT8)

    def simple(self, imgs):
        """Implement simpler method, better if usable when briss is not."""
        img = numpy.square(numpy.average(imgs, axis=0) / 255)
        img = numpy.square(img)
        img = img * 110 + 140
        return img.astype(UINT8)


class CropFinder(object):
    """Define auto-crop interface."""

    def find(self, img):
        pass


class BrissCropFinder(CropFinder):
    """Implement briss's auto-crop method (in CropFinder.java).

    https://github.com/fatso83/briss-archived
    """

    STD_SIZE = 5  # SD_CALC_SIZE_NR
    SIZE = 30  # LOOK_AHEAD_PIXEL_NR
    THRESHOLD = 0.2  # SD_THRESHOLD_TO_BE_COUNTED
    SUCCESS_RATE = 0.85  # RATIO_LOOK_AHEAD_SATISFY

    def find(self, img):
        shape = img.shape

        X = self._sum_x(img)
        Y = self._sum_y(img)

        X = self._diff(X)
        Y = self._diff(Y)

        wx, wy = self._get_wsize(self.STD_SIZE, shape)
        X = self._std(self._roll(X, wx), wx)
        Y = self._std(self._roll(Y, wy), wy)

        wx, wy = self._get_wsize(self.SIZE, shape)
        x0, x1 = self._find(self._roll(X, wx), wx)
        y0, y1 = self._find(self._roll(Y, wy), wy)

        return x0, y0, x1, y1

    def _get_wsize(self, desired, shape):
        # Crip window size to at most 20% of length.
        x = min(desired, max(int(0.2 * shape[1]), 1))
        y = min(desired, max(int(0.2 * shape[0]), 1))
        return x, y

    def _sum_x(self, array):
        return numpy.sum(array, axis=0) / len(array)

    def _sum_y(self, array):
        return numpy.sum(array, axis=1) / len(array[0])

    def _diff(self, array):
        # return numpy.diff(array)
        return numpy.diff(array, append=0)

    def _roll(self, array, window_size):
        """Generate consecutive, overlapping subsets of array.

        https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html

        c.f.
        Numpy v1.20 adds safer '.sliding_window_view'.

        It is for C-style arrays (not Fortran order).
        https://stackoverflow.com/a/6811241 (James Mchugh's comment)
        """
        as_strided = numpy.lib.stride_tricks.as_strided
        wsize = min(window_size, array.shape[-1])
        shape = array.shape[:-1] + (array.shape[-1] - wsize + 1, wsize)
        strides = array.strides + (array.strides[-1],)
        return as_strided(array, shape=shape, strides=strides)

    def _std(self, array, wsize):
        # return numpy.std(array, axis=1)
        return numpy.pad(numpy.std(array, axis=1), wsize)

    def _find(self, array, wsize):
        min_ = 0
        max_ = len(array) - 1

        array = numpy.count_nonzero(array > self.THRESHOLD, axis=1)
        array = numpy.nonzero(array > (wsize * self.SUCCESS_RATE))[0]

        if len(array) == 0:
            return min_, max_

        min_ = max(array.min() - 1, min_)  # min - 1
        max_ = min(array.max() + wsize, max_)  # max + 1
        return int(min_), int(max_)  # int: numpy to Python


class Backend(object):
    """Define external pdf library interface."""

    def __init__(self, fname):
        self.fname = fname
        self.mediaboxes = []
        self.cropboxes = []

    def get_img(self, number):
        pass

    # optional: only used in interpreter ``do_info``.
    def print_info(self, numbers, printout, opt=None):
        printout('Not implemented.')

    # Each backend decides how to handle when 'is_single_boxes' is False
    # (when a page has multiple boxes).
    def write(self, numbers, outfile, is_single_boxes=True):
        pass


class _PyMuPDFBackend(Backend):
    """Handle the library's version differences."""

    def __init__(self, *args, **kwargs):
        if not fitz:
            raise ImportError('Failed to import PyMuPDF (fitz).')
        super().__init__(*args, **kwargs)

    # handle depreciated names
    # https://pymupdf.readthedocs.io/en/latest/znames.html
    def _compat(self, *names):
        def wrapper(obj):
            for name in names:
                attr = getattr(obj, name, None)
                if attr:
                    break
            return attr
        return wrapper

    # v1.18.13 introduced list type argument
    def _delete_pages(self, pdf, indices):
        try:
            pdf.delete_pages(indices)
        except TypeError:
            for n in reversed(indices):
                pdf.delete_page(n)

    # The same arguments as fitz's '.ez_save' (v1.18.11) except 'garbage'
    def _save(self, pdf, outfile, args=None):
        kwargs = dict(
            garbage=2,  # ez_save has 'garbage=3'
            clean=False,
            deflate=True,
            incremental=False,
            ascii=False,
            expand=False,
            linear=False,
            pretty=False,
            encryption=1,  # PDF_ENCRYPT_NONE
            permissions=4095,  # permits all
            owner_pw=None,
            user_pw=None,
        )

        annotations = fitz.Document.save.__annotations__
        if 'deflate_images' in annotations:
            new = dict(
                deflate_images=True,
                deflate_fonts=True,
            )
            kwargs.update(new)

        if 'no_new_id' in annotations:  # from v1.19.0
            new = dict(
                no_new_id=True,
            )
            kwargs.update(new)

        if args:
            kwargs.update(args)

        pdf.save(outfile, **kwargs)


class PyMuPDFBackend(_PyMuPDFBackend):
    """Implement ``Backend`` using PyMuPDF."""

    def __init__(self, *args, **kwargs):
        pdf_obj = kwargs.pop('pdf_obj', None)
        super().__init__(*args, **kwargs)

        self._password = None  # keep password as plaintext
        if pdf_obj:
            self.pdf = pdf_obj
        else:
            self.pdf = self.load_pdf()

        self._info_error = False
        self.imgboxes = self.get_imgboxes()
        self.data = self.get_data()

        self.mediaboxes = tuple(imgbox.mediabox for imgbox in self.imgboxes)
        self.cropboxes = tuple(imgbox.cropbox for imgbox in self.imgboxes)
        self.remove_cropboxes()

        self._cache = {}  # img cache

    def load_pdf(self):
        doc = fitz.open(self.fname)
        return self.decrypt(doc)

    def decrypt(self, doc):
        is_encrypted = self._compat('is_encrypted', 'isEncrypted')
        if self._password:
            doc.authenticate(self._password)
        if is_encrypted(doc):
            doc.authenticate('')
        cnt = 0
        first = True
        password = None
        while is_encrypted(doc):
            if first is True:
                print('The document is password protected. '
                    'Will abort after three unsuccessful inputs.')
                first = False
            else:
                cnt += 1
                if cnt < 3:
                    print('Wrong password. Try again.')
                else:
                    print('Authentication failed. Exiting...')
                    sys.exit(1)

            password = input('Enter Password:').strip()
            doc.authenticate(password)

        if password is not None:
            self._password = password
        return doc

    def get_imgboxes(self):
        imgboxes = []
        for mbox, cbox, rotation in self._get_imgboxes_data():
            imgboxes.append(_PyMuPDFImgBox(mbox, cbox, rotation))
        return imgboxes

    def _get_imgboxes_data(self):
        keys = {
            'mbox': self._compat('mediabox', 'MediaBox'),
            'cbox': self._compat('cropbox', 'CropBox'),
            'rotation': self._compat('rotation',),
        }

        for page in self.pdf:
            mbox = tuple(keys['mbox'](page))
            cbox = tuple(keys['cbox'](page))
            rotation = keys['rotation'](page)
            yield mbox, cbox, rotation

    # c.f. since v1.19.4, 'set_mediabox' removes cropbox,
    # so it can be used instead.
    def remove_cropboxes(self):
        set_cropbox = self._compat('set_cropbox', 'setCropBox')
        for page, imgbox in zip(self.pdf, self.imgboxes):
            set_cropbox(page)(imgbox.mbox2cbox)

    def get_data(self):
        data = {}
        data['info'] = {}
        data['info']['doc'] = {}
        data['info']['doc']['labels'] = {}
        data['info']['_pages'] = {}  # raw pdf boxes data
        data['info']['pages'] = {}  # pdfslash boxes data

        if getattr(self.pdf, 'xref_get_keys', None) is None:  # v1.18.7
            return data

        def eprint(e):
            fmt = 'Error while parsing PDF data, %s: %s'
            print(fmt % (type(e).__name__, str(e)))

        try:
            self._get_doc_info(data)
        except Exception as e:
            eprint(e)
        try:
            self._get_raw_page_info(data)
        except Exception as e:
            eprint(e)
        try:
            self._get_page_info(data)
        except Exception as e:
            eprint(e)

        return data

    def _get_doc_info(self, data):
        self._get_labels(data)

    def _get_labels(self, data):
        rules = self.pdf.get_page_labels()  # v1.18.7
        if not rules:
            return

        rules.sort(key=lambda x: x['startpage'])
        startpages = [rule['startpage'] for rule in rules]
        startpages.append(len(self.pdf))

        i = 0
        labels = []
        for n in range(1, len(self.pdf) + 1):
            if n > startpages[i + 1]:
                i += 1
            rule = rules[i]

            startpage = rule['startpage']
            prefix = rule['prefix']
            style = rule.get('style', '')
            firstpagenum = rule['firstpagenum']

            pagenumber = n - startpage + firstpagenum - 1
            if prefix == '' and style == 'D':
                labels.append(pagenumber)  # adding int as is
            else:
                label = fitz.utils.construct_label(style, prefix, pagenumber)
                if label == '':
                    label = '(none)'
                labels.append(label)

        data['info']['doc']['labels'] = labels

    # c.f. v1.19.4 adds direct APIs for other boxes, e.g. Page.bleedbox
    def _get_raw_page_info(self, data):
        bboxes = 'MediaBox', 'CropBox', 'BleedBox', 'TrimBox', 'ArtBox'
        others = 'Rotate', 'UserUnit'
        info = {}
        for name in bboxes + others:
            info[name] = []

        for page in self.pdf:
            keys = self.pdf.xref_get_keys(page.xref)
            seen = set()
            for name in bboxes:
                vals = info[name]
                if name in keys:
                    _, box = self.pdf.xref_get_key(page.xref, name)
                    if box and box != 'null' and box not in seen:
                        seen.add(box)
                        vals.append(box)
                        continue
                vals.append(None)

            for name in others:
                vals = info[name]
                if name in keys:
                    _, attr = self.pdf.xref_get_key(page.xref, name)
                    if attr and attr != 'null':
                        vals.append(attr)
                        continue
                vals.append(None)

        data['info']['_pages'] = info

    def _get_page_info(self, data):
        bboxes = 'mediabox', 'cropbox', 'bleedbox', 'trimbox', 'artbox'
        others = 'rotate', 'userunit'
        info = {}
        for name in bboxes + others:
            info[name] = []

        for i in range(len(self.pdf)):
            mediabox = self.imgboxes[i].mediabox
            info['mediabox'].append(mediabox)
            seen = set((mediabox,))

            cropbox = self.imgboxes[i].cropbox
            if cropbox not in seen:
                seen.add(cropbox)
                info['cropbox'].append(cropbox)
            else:
                info['cropbox'].append(None)

            for name in ('BleedBox', 'TrimBox', 'ArtBox'):
                vals = info[name.lower()]
                box = data['info']['_pages'][name][i]
                if box is not None:
                    box = self.imgboxes[i].pdfbox2box(box)
                    if box not in seen:
                        seen.add(box)
                        vals.append(box)
                        continue

                vals.append(None)

            rot = self.imgboxes[i].rot
            info['rotate'].append(rot)

            for name in ('UserUnit',):
                vals = info[name.lower()]
                val = data['info']['_pages'][name][i]
                vals.append(val)  # val may be 'None'

        data['info']['pages'] = info

    def print_info(self, numbers, printout, opt=None):
        printout('Page Count: %s' % len(numbers))

        ret = self._format_labels(numbers)
        if ret:
            printout('Page Labels: %s' % ret)

        ret = self._format_page_attrs(numbers, opt)
        if ret:
            printout(ret)

    def _format_labels(self, numbers):
        labels = self.data['info']['doc']['labels']
        if not labels:
            return ''

        labels = [labels[n - 1] for n in numbers]
        ret = []
        stack = []
        for label in labels + ['zzzzz']:  # add one iter to handle the last
            if isinstance(label, int):
                stack.append(label)
            else:
                if stack:
                    ret.append(g_numparser.unparse(stack))
                    stack = []
                ret.append(label)

        return ', '.join(ret[:-1])

    def _format_page_attrs(self, numbers, opt):
        if self._info_error:
            return

        if opt == 'pdf':
            info = self.data['info'].get('_pages')
        else:
            info = self.data['info'].get('pages')

        if not info:
            return ''

        ret = []
        for name, values in info.items():
            first = True
            key = lambda x: values[x - 1]
            for groups in groupby(numbers, key=key):
                attr, nums = groups
                if attr is None:
                    continue
                if first:
                    ret.append('%s:' % name)
                    first = False
                nstr = g_numparser.unparse(nums)
                ret.append('    %-30s  (%s)' % (attr, nstr))

        return '\n'.join(ret)

    def get_img(self, number):
        cache = self._cache.get(number)
        if cache is None:
            cache = self._cache[number] = self._get_img(number)
        return cache

    def _get_img(self, number):
        index = number - 1
        page = self.pdf[index]
        width, height = self.mediaboxes[index][2:]
        clip = (0, 0, width, height)  # clipping them to ints
        get_pixmap = self._compat('get_pixmap', 'getPixmap')(page)
        bytes_ = get_pixmap(
            colorspace='gray', alpha=False, clip=clip, annots=False).samples
        array = numpy.frombuffer(bytes_, dtype=UINT8)
        array.shape = (height, width)
        return array

    def write(self, numbers, boxes, outfile, args, is_single_boxes=True):
        indices = num2ind(numbers)
        _time('start')
        pdf = self.load_pdf()  # creating new pdf object
        _time('(write) PDF reload')

        if is_single_boxes:
            if len(indices) != len(pdf):
                _time('start')
                pdf.select(indices)
                _time('(write) page select, %d pages' % len(indices))
        else:
            _time('start')
            self._copy_pages(pdf, indices)  # deep copy
            _time('(write) page copy and select, %d pages' % len(indices))
            self._adjut_toc(pdf, indices, boxes)

        set_cropbox = self._compat('set_cropbox', 'setCropBox')

        _time('start')
        for i, index in enumerate(indices):
            if boxes[i] is None:
                continue
            page = pdf[i]
            box = self.imgboxes[index].cropbox2cbox(boxes[i])
            set_cropbox(page)(box)
        _time('(write) set cropboxes')

        _time('start')
        self._save(pdf, outfile, args)
        _time('(write) save (fitz.save)')
        pdf.close()

    def _copy_pages(self, pdf, indices):
        length = len(pdf)
        excluded = [i for i in range(length) if i not in indices]
        if excluded:
            self._delete_pages(pdf, excluded)
        prev = -1
        for i, index in enumerate(indices):
            if index == prev:
                if index < length - 1:
                    pdf.fullcopy_page(i - 1, i)
                else:
                    pdf.fullcopy_page(i - 1)
            prev = index

    def _adjut_toc(self, pdf, numbers, boxes):
        pass


class _PyMuPDFImgBox(object):
    """Translate PyMuPDF's box values."""

    def __init__(self, mbox, cbox, rotation):
        self.mbox = mbox
        self.cbox = cbox
        rot = self.rot = rotation % 360
        if rot not in (0, 90, 180, 270):
            raise ValueError('Invalid rotation: %g' % rot)

    def ints(self, box):
        w, h = getsize(box)
        x0, y0 = box[:2]
        x0, y0 = math.ceil(x0), math.ceil(y0)
        return x0, y0, x0 + w, y0 + h

    @property
    def size(self):
        return getsize(self.mbox)

    def rotate(self, box):
        return rotate(*self.size, self.rot, box)

    def unrotate(self, box):
        return unrotate(*self.size, self.rot, box)

    @property
    def mediabox(self):
        w, h = self.size
        if self.rot in (0, 180):
            return 0, 0, w, h
        else:  # 90 or 270
            return 0, 0, h, w

    @property
    def cropbox(self):
        box = shift_box(self.cbox, (-self.mbox[0], 0) * 2)
        box = self.ints(box)
        box = self.rotate(box)
        return box

    def cropbox2cbox(self, box):
        box = self.unrotate(box)
        box = shift_box(box, (self.mbox[0], 0) * 2)
        return self._adjust_cbox(box)

    def _adjust_cbox(self, box):
        # moving bottom remainder to top, trying to get integer PDF values
        # >>> math.modf(-5.5)
        # (-0.5499999999999998, -5.0)
        mbox = self.mbox
        remainder = abs(math.modf(mbox[3] - mbox[1])[0])
        pos = 0, remainder
        return shift_box(box, pos * 2)

    @ property
    def mbox2cbox(self):
        return shift_box(self.mbox, (0, -self.mbox[1]) * 2)

    def pdfbox2box(self, box):  # parse other PDF box string (TrimBox etc.)
        box = pdfbox2box(box)
        box = self._shift_pdfbox(box)
        return box

    def _shift_pdfbox(self, box):
        box = box[0], box[3], box[2], box[1]
        box = shift_box(box, (-self.mbox[0], -self.mbox[1]) * 2)
        box = self.ints(box)
        return box


class Document(object):
    """Manage page and img objects."""

    SUFFIX = '.slashed'

    def __init__(self, fname, conf,
            backend=None,
            numparser=None,
            boxparser=None,
            imgmerger=None,
            cropfinder=None,
            pdf_obj=None):

        self.fname = fname
        self.conf = conf

        backend = backend or PyMuPDFBackend
        self.backend = backend(fname, pdf_obj=pdf_obj)

        boxes = self.backend.mediaboxes, self.backend.cropboxes
        self.pages = _Pages(*boxes)

        self.imgs = _ImgGroup(self, *boxes)

        numparser = numparser or NumParser
        self.numparser = numparser(len(self.pages))

        boxparser = boxparser or BoxParser
        self.boxparser = boxparser(self.pages)

        imgmerger = imgmerger or ImgMerger
        self.imgmerger = imgmerger()

        cropfinder = cropfinder or BrissCropFinder
        self.cropfinder = cropfinder()

        # ``_ImageData`` uses this cache dict.
        self._img_cache = {}

    def free(self):
        self.imgmerger._cache = {}
        self.backend._cache = {}
        self._img_cache = {}

    def print_conf_options(self):
        conf_self = self.conf['_self']
        conf_self.print_items(self.conf)

    def set_conf_option(self, key, val):
        if key == 'merge':
            if val != self.conf['merge']:
                self._img_cache = {}

        conf_self = self.conf['_self']
        conf_self.set_item(self.conf, key, val)

    def autocrop(self, numbers):
        numbers = self.pages.modifiable(numbers)
        commands = []
        _time('start')
        for num in numbers:
            boxes = self.pages.get_boxes(num, fallback=False)
            if len(boxes) == 1:
                box = boxes[0]
            else:  # when number of box is zero or more than one
                box = self.pages[num].cropbox
            box = self._autocrop(num, box)
            command = 'overwrite', num, box
            commands.append(command)
        self.pages.set_each(commands)
        _time('autocrop, %d pages' % len(numbers))

    def _autocrop(self, num, box):
        img, cnt = self.imgs[num - 1].load()
        img = img[0]
        newimg = self._narrow_view(img, box)
        newbox = self.cropfinder.find(newimg)
        return self._unnarrow_view(newbox, box)

    def _narrow_view(self, img, box):
        """Return a boxed part of image data.

        ``CropFinder`` doesn't know about this.
        """
        left, top, right, bottom = box
        return img[top:bottom + 1, left:right + 1]

    def _unnarrow_view(self, newbox, box):
        """Translate new box in the original img coordinates."""
        left, top, right, bottom = newbox
        x, y = box[:2]
        return x + left, y + top, x + right, y + bottom

    def preview(self, numbers, kind='subgroup', _quit=False):
        # numbers = self.pages.selectable(numbers)
        numbers = self.pages.modifiable(numbers)
        runner = self._get_tkrunner(numbers, kind)
        if _quit:
            runner.quit()
        else:
            runner.run()
            return runner

    def _get_tkrunner(self, numbers, kind='subgroup'):
        indices = num2ind(numbers)
        imagedata = _ImageData(self, indices, kind)
        tkrunner = TkRunner(imagedata, self)
        return tkrunner

    def write(self, numbers, args):
        numbers = self.pages.selectable(numbers)
        nums, boxes = self.pages.get_boxes_flattened(numbers)
        is_single_boxes = self.pages.is_single_boxes(numbers)
        name = self._create_outfilename()
        self.backend.write(nums, boxes, name, args, is_single_boxes)

    def _create_outfilename(self):
        fname = self.fname
        suffix = self.SUFFIX
        root, *ext = fname.rsplit('.', maxsplit=1)
        if ext and ext[0].lower() == 'pdf':
            return root + suffix + '.' + ext[0]
        else:
            return fname + suffix + '.pdf'


class _Rect(object):
    """Define rectangle data attributes."""

    color_types = ('all', 'some', 'none')
    colors = {
        'all': COLORS['blue'],  # box for all pages
        'some': COLORS['lightblue'],  # box for some pages
        'active': COLORS['orange'],  # active box (editable)
        'none': None,  # box for no pages
    }

    def __init__(self, rects, box, gid=None):
        self._rects = rects
        self._box = box
        self.gid = gid  # gui object gid
        self.dash = ()  # normal line in Tkinter
        self._tempbox = None  # temporary box in gui

    @property
    def box(self):
        if self._tempbox:
            return self._rects.clip_box(self._tempbox)
        return self._box

    @box.setter
    def box(self, box):
        if box is None:
            self._tempbox = None
        else:
            x0, y0, x1, y1 = box
            if (x1 - x0 < 1) or (y1 - y0 < 1):
                return
            self._tempbox = self._rects.clip_box(box)

    @property
    def sbox(self):  # current scale applied box
        return self._rects.i._scaling.get_scaled(self.box)

    @property
    def _numbers(self):
        return self._rects._get_numbers(self._box)

    @property
    def numbers(self):  # im_state filtered numbers (all, odds or evens)
        return self._rects.get_numbers(self._box)

    @property
    def active(self):
        return self._rects.active_index == self._box

    @property
    def color_type(self):
        numbers = self.numbers
        if not numbers:
            return 'none'
        elif numbers == self._rects.numbers:
            return 'all'
        else:
            return 'some'

    @property
    def color(self):
        if self.active:
            color_type = 'active'
        else:
            color_type = self.color_type
        return self.colors[color_type]


class _SelRect(_Rect):
    """Define 'sel' rectangle data attributes.

    'sel' is a rectangle currently drawing in gui, not yet cropped.
    """

    def __init__(self, rects, gid=None):
        super().__init__(rects, None, gid=gid)
        self.dash = (4, 4)  # dotted line in Tkinter

    @property
    def _numbers(self):
        if self.box is None:
            return ()
        return self._rects._numbers

    @property
    def numbers(self):
        return self._rects.numbers

    @property
    def active(self):
        return self._rects.active_index == 0


class _CropBoxRect(object):
    """Define cropbox data attributes."""

    def __init__(self, rects, box, gid=None):
        self._rects = rects
        self._box = box  # _box: compatibility with _Rect
        self.box = box
        self.gid = gid
        self.dash = ()

    @property
    def sbox(self):
        return self._rects.i._scaling.get_scaled(self.box)

    @property
    def color(self):
        return COLORS['green']


class _Rects(object):
    """Manage rectangle data."""

    msg_prefix = '[gui] '

    def __init__(self, imagedata):
        self.i = imagedata
        self.pages = self.i._doc.pages
        self.boxdict = self.pages.boxdata.boxdict.rects

        self.rects = {}
        self.sel = _SelRect(self)
        self.active_index = 0
        self.update('')

    def update(self, msg=None):
        if msg is None:
            msg = self.pages.check_msg()
        if msg:  # when None or ''
            print(msg)

        # let's not remove invalids rect during gui invocation.
        # for box in self.rects:
        #     if box not in self.boxdict:
        #         self.rects[box].gid = None
        #         del self.rects[box]

        for box in self.boxdict:
            if box not in self.rects:
                self.rects[box] = _Rect(self, box)

    def unscale_box(self, sbox):
        return self.i._scaling.get_unscaled(sbox)

    def clip_box(self, box):
        x0, y0, x1, y1 = box
        w, h = self.i._width, self.i._height
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        return x0, y0, x1, y1

    @property
    def _numbers(self):
        return self.i.numbers

    @property
    def numbers(self):
        return self.get_numbers()

    def _get_numbers(self, box=None):
        if box is None:
            return self._numbers
        else:
            return tuple(n for n in self.boxdict[box] if n in self._numbers)

    def get_numbers(self, box=None):
        numbers = self._get_numbers(box)
        state = self.i.im_state
        if state == 0:
            return numbers  # all
        elif state == 1:
            return filter_numbers(numbers, 1)  # odds
        elif state == 2:
            return filter_numbers(numbers, 2)  # evens

    @property
    def cropboxes(self):
        if not self._numbers:
            return []
        cropboxes = set(self.pages[n].cropbox for n in self.numbers)
        cropboxes -= set((self.pages[self._numbers[0]].mediabox,))
        return [_CropBoxRect(self, box) for box in cropboxes]

    def format_msg(self, op, numbers, box='', new_box=''):
        msg = self.i._doc.pages.format_msg(op, numbers, box, new_box)
        return '%s%s' % (self.msg_prefix, msg)

    def append(self):  # always from self.sel
        box, numbers = self.sel.box, self.sel.numbers
        msg = self.format_msg('append', numbers, box)
        self.pages.append(numbers, box, msg=msg)
        self.update()
        return self.rects[box]

    def overwrite(self):  # always from self.sel
        box, numbers = self.sel.box, self.sel.numbers
        msg = self.format_msg('overwrite', numbers, box)
        self.pages.overwrite(numbers, box, msg=msg)
        self.update()
        return self.rects[box]

    def modify(self, rect):
        old, new = rect._box, rect.box
        rect.box = None
        msg = self.format_msg('modify', rect.numbers, old, new)
        self.pages.modify(rect.numbers, old, new, msg=msg)
        self.reset_active()
        self.update()
        return self.rects[new]

    def discard(self, rect):
        box, numbers = rect.box, rect.numbers
        msg = self.format_msg('discard', numbers, box)
        self.pages.discard(numbers, box, msg=msg)
        self.update()
        self.reset_active()

    def get_active(self):
        if self.active_index == 0:
            return self.sel
        else:
            return self.rects[self.active_index]

    def reset_active(self):
        self.active_index = 0

    # Note: 'cycle' iterates on rects with numbers,
    # while '__iter__' iterates on rects with numbers *or* gid.

    def cycle(self, reverse=False):
        if self.active_index == 0:
            old = self.sel
        else:
            old = self.rects[self.active_index]

        if not reverse:
            new = self._next()
        else:
            new = self._prev()

        if old == self.sel and new == self.sel:
            return None, None  # error: no rects to cycle

        return old, new

    def _next(self):
        keys = [0] + sorted(self.rects.keys())
        index = keys.index(self.active_index)
        for box in keys[index + 1:]:
            rect = self.rects[box]
            if rect.numbers:
                self.active_index = box
                return self.rects[box]
        self.active_index = 0
        return self.sel

    def _prev(self):
        keys = [0] + sorted(self.rects.keys())
        index = keys.index(self.active_index)
        for box in reversed(keys[:index]):
            rect = self.rects[box]
            if rect.numbers:
                self.active_index = box
                return self.rects[box]
        self.active_index = 0
        return self.sel

    def __iter__(self):
        return (rect for rect in self.rects.values()
            if rect.numbers or rect.gid)

    def undo(self):
        msg = self.pages.undo()
        self.update(msg)
        return msg

    def redo(self):
        msg = self.pages.redo()
        self.update(msg)
        return msg


def scale_img(img, scale):
    # basic nearest-neighbor interpolation
    def get_indices(size, new_size):
        ratio = new_size / size
        return ((numpy.arange(new_size) + 0.5) / ratio).astype(INT)

    new_shape = [max(int(s * scale), 1) for s in img.shape]  # floors
    rows = get_indices(img.shape[0], new_shape[0])
    cols = get_indices(img.shape[1], new_shape[1])
    return img[rows][:, cols]


class _Scaling(object):
    """Helper to scale gui objects."""

    # borrowed from firefox
    _SCALES = (0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.2, 1.33, 1.5, 1.7, 2.0)

    def __init__(self, dev_scale=1.0, scale=1.0):
        self._dev_scale = dev_scale
        self._scale = scale

    @property
    def scale(self):
        return round(self._dev_scale * self._scale, 3)

    def get_img(self, img):  # get new scaled img
        if self.scale == 1.0:
            return img
        return scale_img(img, self.scale)

    def get_scaled(self, coords):  # get scaled coords (point or box)
        if self.scale == 1.0:
            return coords
        return tuple(int(c * self.scale) for c in coords)

    def get_unscaled(self, coords):  # get (close to) original coords
        if self.scale == 1.0:
            return coords
        return tuple(int(c / self.scale) for c in coords)

    # user can set an arbitrary scale,
    # but next zoom-in or zoom-out always aligns it to one of self._SCALES

    def _next(self):
        if self._scale == self._SCALES[-1]:
            return None
        for scale in self._SCALES:
            if scale > self._scale:
                return scale
        return self._SCALES[-1]  # when self._scale was far larger.

    def _prev(self):
        if self._scale == self._SCALES[0]:
            return None
        for scale in reversed(self._SCALES):
            if scale < self._scale:
                return scale
        return self._SCALES[0]

    def _set(self, scale):
        self._scale = scale


_tk_help = """
    -----------------------------------------------------------
    preview help:
    # <Arrow> means Left, Right, Up or Down keys

    mouse:
        left click:     start selection (top-left)
        drag:           expand selection
        release:        end selection (bottom-right)

    keys:
        H:              print this help in terminal
        q:              quit

        <Arrow>:        move top-left point
        Shift+<Arrow>:  move bottom-right point
        Control+<Arrow>:move rectangle
        h, j, k, l:     move rectangle (Left, Donw, Up, Right)

        Return:         crop by present selection (append)
        Shift+Return:   crop by present selection (replace)

        n:              next image group
        p:              previous image group
        v:              cycle view (both, odds or evens)
        V:              cycle view (reverse direction)
        s:              toggle souce cropbox visibility

        a:              cycle active rectangle
        d:              delete active rectangle
        c:              copy active rectangle
        z:              zoom in
        Z:              zoom out
        u:              undo (box operations)
        r:              redo (box operations)

    (when copy is pending):
        left click:     paste copied rectangle
        x:              paste copied rectangle (the same coords)

    -----------------------------------------------------------
""".lstrip('\n')


class TkRunner(object):
    """Run tkinter gui."""

    _title = 'pdfslash'

    move_increment = {
        '': {  # move top-left point (expand or contract)
            'Left': (-1, 0, 0, 0),
            'Right': (1, 0, 0, 0),
            'Up': (0, -1, 0, 0),
            'Down': (0, 1, 0, 0),
        },
        'shift': {  # move bottom-right point (expand or contract)
            'Left': (0, 0, -1, 0),
            'Right': (0, 0, 1, 0),
            'Up': (0, 0, 0, -1),
            'Down': (0, 0, 0, 1),
        },
        'control': {  # move box as a whole
            'Left': (-1, 0, -1, 0),
            'Right': (1, 0, 1, 0),
            'Up': (0, -1, 0, -1),
            'Down': (0, 1, 0, 1),
        },
    }

    def __init__(self, imagedata, doc):
        self.i = imagedata
        self._doc = doc
        self._conf = doc.conf

        self._image_id = None
        self._notices = []
        self._cropbox_state = 'normal'
        self._copied_box = None

        self._start = None
        self._end = None

        _time('start')
        self.build()
        _time('tkinter init')

    def run(self):
        print('running tkinter...',
            "type 'q' to quit, 'H' to see help in terminal")
        self.root.mainloop()

    def build(self):
        root = tk.Tk()
        root.title(self._title)

        frame = tk.Frame(root)
        frame.grid(column=0, row=0, sticky='nwes')

        _info = tk.StringVar()
        _info.set('')
        label = tk.Label(
            frame, anchor='w', height=1, padx=5, textvariable=_info)
        label.grid(column=0, row=0, sticky='we')

        canvas = tk.Canvas(frame, background='black')
        canvas.grid(column=0, row=1, sticky='nwes')

        canvas.bind("<Button-1>", self._set_start)
        canvas.bind("<B1-Motion>", self._set_selection)
        canvas.bind("<ButtonRelease-1>", self._set_end)

        root.bind('<H>', self.help)
        root.bind('<q>', self.quit)

        root.bind('<Left>', self._move1)
        root.bind('<Right>', self._move1)
        root.bind('<Up>', self._move1)
        root.bind('<Down>', self._move1)
        root.bind('<Shift-Left>', self._move1)
        root.bind('<Shift-Right>', self._move1)
        root.bind('<Shift-Up>', self._move1)
        root.bind('<Shift-Down>', self._move1)

        root.bind('<h>', self._move2)
        root.bind('<j>', self._move2)
        root.bind('<k>', self._move2)
        root.bind('<l>', self._move2)

        root.bind('<Return>', self._crop)
        root.bind('<Shift-Return>', self._crop)

        root.bind('<n>', self._next)
        root.bind('<p>', self._prev)

        root.bind('<v>', self._cycle_view)
        root.bind('<V>', self._cycle_view)

        root.bind('<s>', self._toggle_cropboxes)

        root.bind('<a>', self._cycle_rect)
        root.bind('<d>', self._remove)

        root.bind('<c>', self._copy_box)
        root.bind('<x>', self._paste_box_from_key)

        root.bind('<z>', self._zoom)
        root.bind('<Z>', self._zoom)

        root.bind('<u>', self._undo)
        root.bind('<r>', self._redo)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=0)
        frame.rowconfigure(1, weight=1)

        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()

        self.root = root
        self.frame = frame
        self._info = _info
        self.label = label
        self.canvas = canvas

        self._next()

    def _get_modifier(self, event):
        modifier = getattr(event, 'state', None)
        if modifier & 0x0001:
            return 'shift'
        elif modifier & 0x0004:
            return 'control'
        else:
            return ''

    def quit(self, event=None):
        self._remove_notifications()
        self.root.destroy()

    def help(self, event=None):
        print(_tk_help)

    def _next(self, event=None):
        if self._image_id and self._is_single_group():
            self._notify('only one group.')
            return
        self.i.next()
        self._get_image()

    def _prev(self, event=None):
        if self._is_single_group():
            self._notify('only one group.')
            return
        try:
            self.i.prev()
        except LookupError:
            msg = "can't rewind to the last image group until once reached."
            self._notify(msg)
            return
        self._get_image()

    def _is_single_group(self):
        return self.i.g_num == 1

    def _get_image(self):
        self.canvas.config(width=self.i.width, height=self.i.height)

        if self._image_id is None:
            self._image_id = self.canvas.create_image(
                0, 0, image=self.i.image, anchor='nw', tags='pdfimage')
        else:
            self.canvas.itemconfig(self._image_id, image=self.i.image)

        self._draw()

    def _draw(self):
        self._position_window()
        self._draw_rects()
        if self._sel.gid and self._sel.box:
            self._draw_rect(self._sel)
        self._draw_cropboxes()
        self._set_info()

    def _position_window(self):
        w, h, x, y = self._get_winpos()
        geometry = '%dx%d+%d+%d' % (w, h, x, y)
        self.root.geometry(geometry)
        # NG: cutting leading words rather than trailing words
        # self.label.config(wraplength=width)

    def _get_winpos(self):
        self.root.update_idletasks()
        # w, h = self.root.winfo_width(), self.root.winfo_height()  # NG
        w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        x_ratio, y_ratio = self._conf['winpos']
        x = max(0, (self.screen_w - w)) * x_ratio
        y = max(0, (self.screen_h - h)) * y_ratio
        return w, h, int(x), int(y)

    @property
    def _sel(self):
        return self.i.rects.sel

    def _reset_sel(self):
        self._sel.box = None
        self._draw_rect(self._sel)

    def _draw_rects(self):
        for rect in self.i.rects:
            self._draw_rect(rect)

    def _draw_rect(self, rect):
        if rect.box is None:  # only sel
            if rect.gid is not None:
                self._hide_rect(rect.gid)
            return

        if rect.gid is None:
            if rect.numbers:
                if rect == self._sel:
                    self._create_rect(rect, tag='sel')
                else:
                    self._create_rect(rect, tag='rect')
        else:
            if rect.numbers:
                self._configure_rect(rect)
            else:
                self._hide_rect(rect.gid)

    def _create_rect(self, rect, tag='rect'):
        rect.gid = self.canvas.create_rectangle(
            *rect.sbox, fill='', dash=rect.dash, outline=rect.color,
            state='normal', tags=tag)

    def _configure_rect(self, rect):
        self.canvas.itemconfig(
            rect.gid, dash=rect.dash, outline=rect.color, state='normal')
        self.canvas.coords(rect.gid, *rect.sbox)

    # Note the argument is 'gid'. Also can un-hide.
    def _hide_rect(self, gid, new_state='hidden'):
        self.canvas.itemconfig(gid, state=new_state)

    def _move_rect(self, rect, box):
        rect.box = box
        self.canvas.coords(rect.gid, *rect.sbox)

    def _draw_cropboxes(self):
        for gid in self.canvas.find_withtag('cropbox'):
            self.canvas.delete(gid)
        for rect in self.i.rects.cropboxes:
            self._create_rect(rect, tag='cropbox')

        if self._cropbox_state == 'hidden':
            self._cropbox_state = 'normal'
            self._toggle_cropboxes()

    def _toggle_cropboxes(self, event=None):
        if self._cropbox_state == 'normal':
            new_state = 'hidden'
        else:
            new_state = 'normal'

        for gid in self.canvas.find_withtag('cropbox'):
            self._hide_rect(gid, new_state=new_state)

        self._cropbox_state = new_state

    def _set_start(self, event):
        if self.i.rects.active_index != 0:
            self.i.rects.reset_active()
            self._draw_rects()

        if self._copied_box:
            self._paste_box_from_click(event)
            return

        self._start = event.x, event.y
        if self._sel.box:
            self._reset_sel()

    def _set_selection(self, event):
        if self._copied_box:
            return

        sbox = *self._start, event.x, event.y
        box = self.i.rects.unscale_box(sbox)
        if not self._sel.box:
            self._sel.box = box
            self._draw_rect(self._sel)
        else:
            self._move_rect(self._sel, box)
        self._set_info()

    def _set_end(self, event):
        if self._copied_box:
            self._copied_box = None
            return

        x, y = self._start
        minimum = int(5 * self.i._scaling.scale)
        if (event.x - x) < minimum or (event.y - y) < minimum:
            if self._sel.box:
                self._reset_sel()
            self._set_info()
            return

        self._set_selection(event)

    def _move1(self, event):
        mod = self._get_modifier(event)
        self._move(mod, event.keysym)

    def _move2(self, event):
        mod = 'control'
        keydict = {'h': 'Left', 'j': 'Down', 'k': 'Up', 'l': 'Right'}
        key = keydict[event.keysym]
        self._move(mod, key)

    def _move(self, mod, key):
        rect = self.i.rects.get_active()
        if rect.box is None:  # when self._sel is active and no tempbox
            self._notify('no selection')
            return

        pos = self.move_increment[mod][key]
        box = shift_box(rect.box, pos)
        self._move_rect(rect, box)
        self._set_info()

    def _crop(self, event):
        if not self.i.numbers:  # no page, gui is showing a black image
            self._notify('no page')
            return

        rect = self.i.rects.get_active()
        modifier = self._get_modifier(event)
        append = False if modifier == 'shift' else True

        if rect.box is None:  # when self._sel is active and no tempbox
            self._notify('no selection')
            return

        if rect == self._sel:
            if append:
                rect = self.i.rects.append()
                self._draw_rect(rect)
            else:
                rect = self.i.rects.overwrite()
                self._draw_rects()
            self._reset_sel()
        else:
            if rect.box == rect._box:  # when rect is not moved
                self._notify('not modified')
                return

            newrect = self.i.rects.modify(rect)
            self._draw_rect(rect)
            self._draw_rect(newrect)

        self.i._set()
        self._set_info()

    def _remove(self, event):
        rect = self.i.rects.get_active()
        if rect == self._sel:
            if self._sel.box is None:
                self._notify('no box to remove')
            else:
                self._reset_sel()
            return

        self.i.rects.discard(rect)
        self._draw_rect(rect)
        self.i._set()
        self._set_info()

    def _copy_box(self, event):
        rect = self.i.rects.get_active()
        if rect.box is None:
            self._notify('no active box to copy')
            return
        self._copied_box = rect.box
        self._set_title()

    def _paste_box_from_click(self, event):
        x, y = event.x, event.y
        x, y = self.i._scaling.get_unscaled((x, y))
        w, h = getsize(self._copied_box)
        box = x, y, x + w, y + h
        self._paste_box(box)

    def _paste_box_from_key(self, event):
        if not self._copied_box:
            return
        if self.i.rects.active_index == self._copied_box:
            self._notify("can't paste in the same pages")
            return
        self._paste_box()

    def _paste_box(self, box=None):
        box = box or self._copied_box
        self._sel.box = box
        self._draw_rect(self._sel)
        # if from click, wait drag and release, then to 'None'
        self._copied_box = 'done' if box else None
        self._set_info()

    def _cycle_view(self, event):
        if event.keysym == 'v':
            inc = 1
        elif event.keysym == 'V':
            inc = -1
        im_state = (self.i.im_state + inc) % 3  # c.f. -1 % 3 = 2
        self.i.update(im_state=im_state)
        self._get_image()

    def _cycle_rect(self, event):
        old, new = self.i.rects.cycle()
        if old is None and new is None:  # error
            self._notify('cannot cycle rect (no rect)')
            return
        for rect in (old, new):
            # if rect.box is not None:
            self.canvas.itemconfig(rect.gid, outline=rect.color)
        self._set_info()

    def _zoom(self, event):
        if event.keysym == 'z':
            which = 'zoom_in'
        elif event.keysym == 'Z':
            which = 'zoom_out'
        self.i._zoom(which)
        self._get_image()

    def _undo(self, event):
        try:
            self.i.undo()
        except UndoError as e:
            msg = e.tostring()
            print(msg)
            self._notify(msg)
            return
        self._reset_sel()
        self._get_image()

    def _redo(self, event):
        try:
            self.i.redo()
        except RedoError as e:
            msg = e.tostring()
            print(msg)
            self._notify(msg)
            return
        self._reset_sel()
        self._get_image()

    def _set_info(self):
        self._set_title()
        self._set_label()

    def _set_title(self):
        pages = self.i.nstr if self.i.numbers else '(none)'

        if self.i._scaling.scale == 1.0:
            scale = ''
        else:
            scale = ' (%g%%)' % (self.i._scaling.scale * 100)

        if self._copied_box in (None, 'done'):
            copy = ''
        else:
            copy = ' [copy]'

        self.root.title('%s: %s%s%s' % (self._title, pages, scale, copy))

    def _set_label(self):
        group = self.i.g_index + 1, self.i.g_num
        states = ['both', 'odds', 'evens']
        imstate = states[self.i.im_state]
        size, box = None, None
        if self.i.numbers:
            size = self.i._width, self.i._height
            box = self.i.rects.get_active().box
        text = self._format_info(
            group=group, imstate=imstate, size=size, box=box)
        self._info.set(text)

    def _format_info(self, group=None, imstate=None, size=None, box=None):
        def add_space(text, comma=True):
            if text:
                text += ',  ' if comma else ' '
            return text

        text = ''
        if group:
            text = '%d/%d' % group
        if imstate:
            text = add_space(text, False) + '%s' % imstate
        if size:
            text = add_space(text, False) + '%dx%d' % size

        if self.i.rects.active_index == 0:
            name = 'sel'
            if not box:
                return add_space(text) + '%s: (none)' % name
        else:
            name = 'box'

        bsize = getsize(box)
        ratio = 0 if bsize[0] == 0 else bsize[1] / bsize[0]
        fmt = '%s: %d,%d,%d,%d (%dx%d, %.3f)'
        text = add_space(text) + fmt % (name, *box, *bsize, ratio)
        return text

    def _notify(self, text, duration=2000):
        self._remove_notifications()
        x = self.i.width // 2
        y = self.i.height // 2
        fontsize = int(12 * self.i._scaling.scale)
        font = ('TkTextFont', fontsize)
        t = self.canvas.create_text(x, y,
            fill=COLORS['red'], font=font, text=text, tags='notice')
        n = self.canvas.after(duration, self.canvas.delete, t)
        self._notices.append(n)

    def _remove_notifications(self):
        for n in self._notices:
            self.canvas.after_cancel(n)
        for handle in self.canvas.find_withtag('notice'):
            self.canvas.delete(handle)
        self._notices = []

    def _notify2(self, rect):  # not used
        def _restore():
            self.canvas.itemconfig(rect.gid, width=1)
        self.canvas.itemconfig(rect.gid, width=2)
        self.canvas.after(300, _restore)


class NumParser(object):
    """Parse and unparse number string (nstr).

    Spec:

    1-5         1 to 5 inclusive (1,2,3,4,5)
    3-          3 to last page
    -10         1 to 10

    1^5         every other page in 1 to 5 inclusive (1,3,5)
    2^6         every other page in 2 to 6 inclusive (2,4,6)
                two operands must be both odds or both evens.

    :           all pages

    ~           the same pages as the previous command
    """

    def __init__(self, length):
        self.length = length
        self._prev = None

    def _error_fmt(self, supl=''):
        fmt = 'Invalid number string: %s'
        if supl:
            fmt = '%s (%s)' % (fmt, supl)
        return fmt

    def parse(self, nstr):
        """Create numbers list from ``nstr``.

        e.g. [3, 5, 6, 7, 18, 19, ... <last page>] from '3,5-7,18-'.
        """
        nstr = nstr.strip()
        if nstr == '':
            return []
        if nstr == ':':
            return list(range(1, self.length + 1))
        if nstr == '~':
            if self._prev is None:
                raise ValueError("no previous numbers for '~'")
            return self._prev

        numbers = []
        nums = [n.strip() for n in nstr.split(',')]
        _error_supl = ''

        if len(nums[0]) > 1 and nums[0].startswith('-'):
            nums[0] = '1' + nums[0]
        if len(nums[-1]) > 1 and nums[-1].endswith('-'):
            nums[-1] += str(self.length)

        def int2(num):
            i = int(num)
            if i < 1:
                _error_supl = '0 or minus numbers'  # noqa F841 never used
                raise ValueError
            return i

        for num in nums:
            try:
                i = -1
                i = num.find('-')
                if i == -1:
                    i = num.find('^')

                if i == -1:
                    numbers.append(int2(num))
                    continue

                n1, n2 = num[:i], num[i + 1:]
                n1, n2 = int2(n1.strip()), int2(n2.strip())
                if n1 >= n2:
                    _error_supl = 'left operand is equal or greater than right'
                    raise ValueError
                if num[i] == '-':
                    lst = list(range(n1, n2 + 1))
                elif num[i] == '^':
                    if (n1 % 2) != (n2 % 2):
                        _error_supl = "'^' got odd and even numbers"
                        raise ValueError
                    lst = list(range(n1, n2 + 1, 2))
                numbers.extend(lst)

            except (ValueError, IndexError):
                raise ValueError(self._error_fmt(_error_supl) % nstr)

        numbers = [n for n in numbers if n <= self.length]  # here, no Error
        numbers = tuple(sorted(numbers))
        self._prev = numbers
        return numbers

    def unparse(self, numbers):
        """Create nstr from sorted numbers list or tuple.

        e.g. '3,5-7,18^22' from [3, 5, 6, 7, 18, 20, 22].
        """
        nstr = []
        stack = []
        state = ''  # one of '', '-' or '^'

        def reset(n):
            nonlocal state, stack
            state = ''
            stack = [n]

        for n in tuple(numbers) + (-99999,):  # add one iter to handle the last
            if not stack:
                reset(n)
                continue

            if state in ('', '-') and n == stack[-1] + 1:
                state = '-'
                stack.append(n)
                continue

            if state in ('', '^') and n == stack[-1] + 2:
                state = '^'
                stack.append(n)
                continue

            if len(stack) == 1:
                nstr.append(str(stack[0]))
                reset(n)
                continue

            if len(stack) == 2:  # [2, 3] to '2,3', not '2-3'
                nstr.append(str(stack[0]))
                s1 = stack[1]
                if n == s1 + 1:
                    state = '-'
                    stack = [s1, n]
                elif n == s1 + 2:
                    state = '^'
                    stack = [s1, n]
                else:
                    nstr.append(str(s1))
                    reset(n)
                continue

            start, end = str(stack[0]), str(stack[-1])
            nstr.append(start + state + end)
            reset(n)

        return ','.join(nstr)


class BoxParser(object):
    """Parse box string (bstr) and do crop.

    Spec:

    10,20,30,40     left,top,right,bottom
                    (they must be integers, no dots).

    For 'modify' command, the following special syntax can be used.

    For box1 (box to modify):

    @               Specify box with order.
                    E.g. '@1' means first boxes for each page.

    For box2 (box to modify *to*):

    +-              Apply increment or decrement
                    to the chosen boxes (by box1) of each page.

                    E.g. when box is '-3,-3,+3,+3':

                        20,20,400,400  ->  17,17,403,403
                        30,30,600,600  ->  27,27,603,603

    min, max        min or max numbers
                    of the chosen boxes (by box1) of each page.

                    E.g. min,min,max,+0
                    (select the broadest rectangle
                     for left, top and right,
                     but do not change the bottoms.)
    """

    def __init__(self, pages):
        self._pages = pages
        self._cache = {
            'numbers': None,
            'pageboxes': None,
        }

    def _initialize(self):
        for key in self._cache:
            self._cache[key] = None

    def _get_pageboxes(self, numbers):
        cache = self._cache
        if cache['pageboxes'] is None:
            pageboxes = self._pages.get_pageboxes(numbers, fallback=False)
            if [] in pageboxes:
                index = pageboxes.index([])
                raise NoBoxToProcessError(index + 1)
            cache['pageboxes'] = pageboxes
        return cache['pageboxes']

    def parse(self, numbers, *args):
        table = (
            ('b', '_parse_b'),
            ('bb', '_parse_bb'),
            ('B', '_parse_B'),
        )

        self._initialize()
        args, signature = args[:-1], args[-1]
        for sig, name in table:
            if signature == sig:
                method = getattr(self, name)
                return method(numbers, *args)

        raise ValueError('Error: Invalid signature: %s.' % signature)

    def _parse_b(self, numbers, bstr):  # append or overwrite
        box = self._get_plain_box(bstr)
        return numbers, box

    def _parse_bb(self, numbers, bstr1, bstr2):  # modify
        box1, boxes1 = self._get_box1(numbers, bstr1)
        box2, boxes2 = self._get_box2(numbers, box1, boxes1, bstr2)
        if box1:
            return 'modify', numbers, box1, box2
        elif boxes1:
            commands = []
            for n, box1, box2 in zip(numbers, boxes1, boxes2):
                command = 'modify', n, box1, box2
                commands.append(command)
            return 'set_each', commands

    def _parse_B(self, numbers, bstr):  # dicard
        box = self._get_plain_box(bstr)
        return numbers, box

    def _get_plain_box(self, bstr):
        try:
            return tuple(int(b) for b in bstr.split(','))
        except ValueError:
            raise ValueError('Invalid box string: %s' % bstr)

    def _get_box1(self, numbers, bstr1):  # boxquerry
        box1, boxes1 = None, None
        parts = bstr1.split('@')
        try:
            if len(parts) == 1:
                box1 = self._get_plain_box(parts[0])
                return box1, boxes1
            elif len(parts) == 2:
                if parts[0] == '':
                    index = int(parts[1]) - 1
                    if index > -1:
                        pageboxes = self._get_pageboxes(numbers)
                        boxes1 = [boxes[index] for boxes in pageboxes]
                        if all(b == boxes1[0] for b in boxes1[1:]):
                            box1 = boxes1[0]
                            boxes1 = None
                        return box1, boxes1

        except (ValueError, IndexError):
            raise ValueError('Invalid box string: %s' % bstr1)

    def _get_box2(self, numbers, box1, boxes1, bstr2):  # boxmod
        box = [b for b in bstr2.split(',')]  # box: box2
        if len(box) != 4:
            fmt = 'More or less than four box coordinates: %r.'
            raise ValueError(fmt % bstr2)

        coords = []
        for i, b in enumerate(box):
            sign = None
            if b[0] in ('+', '-'):
                sign, b = b[0], b[1:]
            elif b in ('min', 'max'):
                b = self._get_min_max(numbers, box1, boxes1, i, b)
            b = int(b)
            coords.append((sign, b))

        box2, boxes2 = None, None

        if box1:
            box2 = self._apply_plusminus(box1, coords)
        elif boxes1:
            boxes2 = [self._apply_plusminus(b, coords) for b in boxes1]

        return box2, boxes2

    def _get_min_max(self, numbers, box1, boxes1, pos, which):
        if box1:
            return box1[pos]
        elif boxes1:
            values = [box[pos] for box in boxes1]
            funcs = {'min': min, 'max': max, }
            return funcs[which](values)

    def _apply_plusminus(self, box, coords):
        box = list(box)
        for i, (sign, b) in enumerate(coords):
            if sign == '+':
                box[i] += b
            elif sign == '-':
                box[i] -= b
            else:
                box[i] = b
        return tuple(box)


class CommandParser(object):
    """Parse Command on interpreter."""

    def __init__(self, cmd):
        self.numparser = cmd.numparser
        self.boxparser = cmd.boxparser
        self.printout = cmd.printout

    def parse_opts(self, args_):
        opts, args = [], []
        maybe_opts = True
        for a in args_.split():
            if a == '--':
                maybe_opts = False
                continue
            if maybe_opts:
                if self._is_short_opt(a):
                    opts.append(a)
                    continue
            args.append(a)
        return opts, args

    def _is_short_opt(self, tok):
        digits = list('0123456789')  # ['0', '1', '2', ...]
        if tok.startswith('-') and len(tok) > 1 and tok[1] not in digits:
            return True
        return False

    def parse(self, args, signature='n', allow_blank=False):
        opts, args = self.parse_opts(args)
        if not args:
            if allow_blank:
                args = [':']
            else:
                self.printout('Error: No page numbers.')
                return

        ret = self._parse(args, signature)
        return ret, opts  # ret is 'None' or something.

    def _parse(self, args, signature):
        table = (
            ('n', '_parse_n'),
            ('nb', '_parse_nb'),
            ('nbb', '_parse_nbb'),
            ('nB', '_parse_nB'),
            ('nplus', '_parse_nplus'),
        )

        for sig, name in table:
            if signature == sig:
                method = getattr(self, name)
                return method(args)

        self.printout('Error: Invalid signature: %s.' % signature)

    def _get_numbers(self, nstr):
        try:
            return self.numparser.parse(nstr)
        except ValueError as e:
            self.printout('Error while parsing numbers: %s.' % str(e))
            return

    def _parse_n(self, args):
        if len(args) > 1:
            fmt = 'Error: More than one arguments (numbers): %r.'
            self.printout(fmt % args)
            return

        nstr = args[0]
        return self._get_numbers(nstr)

    def _parse_nplus(self, args):
        if len(args) == 1:
            args = [':'] + args
        numbers = self._get_numbers(args[0])
        return (numbers, *args[1:])

    def _parse_nb(self, args):  # append or overwrite
        if len(args) != 2:
            fmt = ('Error: More or less than two arguments '
                '(numbers and box): %r')
            self.printout(fmt % args)
            return

        nstr, bstr = args
        numbers = self._get_numbers(nstr)
        if numbers:
            try:
                return self.boxparser.parse(numbers, bstr, 'b')
            except ValueError as e:
                self.printout('Error while parsing box: %s' % str(e))
                return

    def _parse_nbb(self, args):  # modify
        if len(args) != 3:
            fmt = ('Error: You have to provide three arguments '
                '(numbers, box1, box2): %r')
            self.printout(fmt % args)
            return

        nstr, bstr1, bstr2 = args
        numbers = self._get_numbers(nstr)
        if numbers:
            try:
                return self.boxparser.parse(numbers, bstr1, bstr2, 'bb')
            except (ValueError, NoBoxToProcessError) as e:
                self.printout('Error while parsing box: %s' % str(e))
                return

    def _parse_nB(self, args):  # discard
        if len(args) != 2:
            fmt = ('Error: More or less than two arguments '
                '(numbers and box): %r')
            self.printout(fmt % args)
            return

        nstr, bstr = args
        numbers = self._get_numbers(nstr)
        if numbers:
            try:
                return self.boxparser.parse(numbers, bstr, 'B')
            except ValueError as e:
                self.printout('Error while parsing box: %s' % str(e))
                return


class _NoErrorCmd(cmd.Cmd):
    """Turn Errors to printout."""

    def onecmd(self, line):
        try:
            ret = super().onecmd(line)
        except (LookupError, ValueError, TypeError):
            traceback.print_exc()
            return False
        return ret


class _PipeStdout(object):
    """Pipe stdout into subprocess if self.command filled."""

    def __init__(self, stdout):
        self._stdout = stdout
        self._output = []
        self.command = None

    def write(self, s):
        self._output.append(s)

    def _run(self):
        s = ''.join(self._output)
        self._output = []

        if self.command:
            command = self.command
            self.command = None

            try:
                subprocess.run(command, input=s, stdout=self._stdout,
                    shell=True, encoding=self._stdout.encoding)
            except OSError:  # TODO: need more Errors
                pass
        else:
            self._stdout.write(s)

    def __getattr__(self, name):
        return getattr(self._stdout, name)


class _PipeCmd(_NoErrorCmd):
    """Pipe stdout into subprocess if a command includes '|'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stdout = self.stdout
        self._pipestdout = _PipeStdout(self.stdout)

    def precmd(self, line):
        return self._wrap_stdout(line)

    def postcmd(self, stop, line):
        self._pipestdout._run()
        self.stdout = self._stdout
        return stop

    def _wrap_stdout(self, line):
        line, command = self._split_on_pipe(line)
        if command:
            self._pipestdout.command = command
            self.stdout = self._pipestdout
        else:
            self.stdout = self._stdout
        return line

    def _split_on_pipe(self, line):
        command = ''
        words = shlex.split(line)
        if '|' in words:
            pipe_index = words.index('|')
            line = shlex.join(words[:pipe_index])
            command = self._join_tokens(words[pipe_index + 1:])
        return line, command

    def _join_tokens(self, tokens):
        # basically the same as ``shlex.join``,
        # but special-cases some tokens.
        line = []
        specials = ('>', '>>', '|')
        for token in tokens:
            if token in specials:
                line.append(token)
            else:
                line.append(shlex.quote(token))
        return ' '.join(line)


_cmd_intro = """
This is pdfslash interactive session.

    Type 'help' or '?' to list commands.
    Type 'exit', 'quit' or send EOF to exit.
        (EOF: <Ctrl-D>, or <Ctrl-Z><Enter> on Windows)
    Specify box as: left,top,right,bottom (e.g. 0,0,100,200).
""".lstrip()


class PDFSlashCmd(_PipeCmd):
    """Run interactive session."""

    intro = _cmd_intro
    prompt = '(pdfslash) '

    hisname = '.history'  # history file name
    hisfile = None  # history file path
    hissize = 1000  # max history file size

    pyhisname = '.python_history'  # Python history file name
    pyhisfile = None  # Python history file path

    export_dir = 'exported'

    def __init__(self, *args, **kwargs):
        doc = kwargs.pop('doc', None)
        if doc is None:
            msg = "argument 'doc' (pdfslash.Document object) is required."
            raise ValueError(msg)
        super().__init__(*args, **kwargs)

        self._doc = doc
        self._pages = doc.pages
        self.numparser = self._doc.numparser
        self.boxparser = self._doc.boxparser
        self.cmdparser = CommandParser(self)

        self.hisfile = self._get_history_file(self.hisname)
        self.pyhisfile = self._get_history_file(self.pyhisname)

    def _get_history_file(self, fname):
        config_dir = self._doc.conf['_config_dir']
        if readline and config_dir:
            h = os.path.join(config_dir, fname)
            if not os.path.isfile(h):
                with open(h, 'w'):  # create empty file
                    pass
            return h

    def _get_export_file(self):
        config_dir = self._doc.conf['_config_dir']
        if not config_dir:
            self.printout('Error: user directory is not defined.')
            return

        export_dir = os.path.join(config_dir, self.export_dir)
        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)
        t = time.strftime('%Y%m%d%H%M%S')
        fname = '%s.%s.txt' % (os.path.basename(self._doc.fname), t)
        return os.path.join(export_dir, fname)

    def _start_readline(self, fname):
        if readline and fname:
            readline.clear_history()
            readline.read_history_file(fname)

    def _end_readline(self, fname):
        if readline and fname:
            readline.set_history_length(self.hissize)
            readline.write_history_file(fname)

    def preloop(self):
        self._start_readline(self.hisfile)

    def postloop(self):
        self._end_readline(self.hisfile)

    def printout(self, string):
        self.stdout.write(str(string))
        self.stdout.write('\n')

    def _printout(self, string):  # no pipe ('|') check version
        self._stdout.write(str(string))
        self._stdout.write('\n')

    def do_select(self, args):
        """
        Take one argument, page numbers.

        ``select`` page numbers.

        Operations are done to only selected pages.
        Initially all pages are selected.

        Use when you don't want to repeat very complex page numbers.

        .. code-block:: none

            unselect :                  # unselect all pages
            select 2-8                  # select pages 2-8
            crop 1-10 100,100,400,400   # crop pages 2-8
            write                       # write pages 2-8
        """
        numbers, opts = self.cmdparser.parse(args)
        if numbers:
            self._pages.select(numbers)

    def do_unselect(self, args):
        """
        Take one argument, page numbers.

        ``unselect`` page numbers.

        See ``select``.
        """
        numbers, opts = self.cmdparser.parse(args)
        if numbers:
            self._pages.unselect(numbers)

    def do_fix(self, args):
        """
        Take one argument, page numbers.

        ``fix`` page numbers.

        Box operations are not done to fixed pages.
        Initially all pages are unfixed.

        Use when you want to make some pages 'done'.

        .. code-block:: none

            crop 2,3 150,150,450,450    # crop pages 2,3
            fix 2,3                     # fix pages 2,3
            crop 2-6 100,100,400,400    # crop pages 4,5,6
            write 2-10                  # write pages 2-10
        """
        numbers, opts = self.cmdparser.parse(args)
        if numbers:
            self._pages.fix(numbers)

    def do_unfix(self, args):
        """
        Take one argument, page numbers.

        ``unfix`` page numbers.

        See ``fix``.
        """
        numbers, opts = self.cmdparser.parse(args)
        if numbers:
            self._pages.unfix(numbers)

    def _append_or_overwrite(self, args, which):
        ret, opts = self.cmdparser.parse(args, signature='nb')
        if ret:
            numbers, box = ret
            op = getattr(self._pages, which)
            try:
                op(numbers, box)
            except Exception as e:  # TODO
                self._printout('Error while processing box: %s' % str(e))
                return

    def do_append(self, args):
        """
        Take two argument, page numbers and box.

        Append box.

        (Add box to specified pages, keeping previously added boxes.)
        """
        self._append_or_overwrite(args, which='append')

    def do_overwrite(self, args):
        """
        Take two argument, page numbers and box.

        Replace box.

        (Add box to specified pages, removing previously added boxes.)
        """
        self._append_or_overwrite(args, which='overwrite')

    def do_modify(self, args):
        """
        Take three argument, page numbers, box1 and box2.

        Modify box.

        (For each page, change pre-existent box (box1) to new box (box2).
        If box1 doesn't exist in any page, it is Error).
        """
        ret, opts = self.cmdparser.parse(args, signature='nbb')
        if ret:
            op = getattr(self._pages, ret[0])
            try:
                op(*ret[1:])
            except Exception as e:  # TODO
                self._printout('Error while processing box: %s' % str(e))
                return

    def do_discard(self, args):
        """
        Take two argument, page numbers and box.

        Delete box.

        (Find the box in each specified page, and remove them.
        If the box doesn't exist in any page, it is Error).
        """
        ret, opts = self.cmdparser.parse(args, signature='nB')
        if ret:
            numbers, box = ret
            try:
                self._pages.discard(numbers, box)
            except Exception as e:  # TODO
                self._printout('Error while processing box: %s' % str(e))
                return

    def do_clear(self, args):
        """
        Take one argument, page numbers.

        Clear boxes.

        (Delete all added boxes in specified pages.
        that is, they will revert to the original source cropboxes).
        """
        numbers, opts = self.cmdparser.parse(args)
        if numbers:
            self._pages.clear(numbers)

    def do_auto(self, args):
        """
        Take one argument, page numbers (optional).

        Auto detect page margins and apply (overwrite) them.
        All previously added boxes are removed.

        If the number of previous boxes is one,
        the detection is done against this box,
        else (the number is zero or two or more),
        the detection is done against source cropbox.
        """
        numbers, opts = self.cmdparser.parse(args, allow_blank=True)
        if numbers:
            self._doc.autocrop(numbers)

    def do_preview(self, args):
        """
        Take one argument, page numbers (optional).

        Run tkinter GUI.

        Options (optional):

        ``-m``, ``--mediabox``:
            Group pages by source mediabox
        ``-c``, ``--cropbox``:
            Group pages first by source mediabox,
            and then by source cropbox (for each mediabox group).
            This is the default.
        ``-s``, ``--single``:
            Group each page in each group, to navigate pages one by one.
        ``-_q``, ``--_quit``:
            Create GUI window and immediately quit (for test).
        """
        numbers, opts = self.cmdparser.parse(args, allow_blank=True)
        kind = 'subgroup'
        _quit = False
        if opts:
            if opts[0] in ('-m', '--mediabox'):
                kind = 'group'
            elif opts[0] in ('-c', '--cropbox'):
                kind = 'subgroup'
            elif opts[0] in ('-s', '--single'):
                kind = 'single'
            elif opts[0] in ('-_q', '--_quit'):
                _quit = True
            else:
                self._printout('Invald option: %s' % opts)
                return

        if numbers:
            self._doc.preview(numbers, kind=kind, _quit=_quit)

    def do_write(self, args):
        """
        Take one argument, page numbers (optional).

        Create new PDF file with specified (or *selected*) pages.

        It uses ``PyMuPDF``'s ``fitz.Document.save`` method,
        with the same arguments as ``fitz.Document.ez_save``,
        except 'garbage=2' (instead of '3').

        Options (optional):

        ``-m``, ``--more``:
            Shortcut for ``-a{'garbage':3}``.
            For shorter PDF, it seems OK. May make file size smaller,
            but it tends to get very slower.
        ``-a``, ``--args``:
            Update the default arguments.
            The string after, say, ``-a`` must be valid Python code,
            evaluating to a dictionary, with no spaces.
        """
        numbers, opts = self.cmdparser.parse(args, allow_blank=True)
        if numbers:
            self._printout('writing...')
            args, code = None, None
            if opts:
                opts = opts[0]
                if opts in ('-m', '--more'):
                    args = {'garbage': 3}
                elif opts[:2] == '-a':
                    code = opts[2:]
                elif opts[:6] == '--args':
                    code = opts[6:]
                if code:
                    args = eval(code)
                    self._printout('adding args, %r' % args)

            self._doc.write(numbers, args=args)

    def do_show(self, args):
        """
        Take one argument, page numbers (optional).

        Show current boxes for specified pages.

        If selected or fixed,
        pages are shown with headers ``'s'`` and ``'f'`` respectively.
        """
        numbers, opts = self.cmdparser.parse(args, allow_blank=True)
        if numbers:
            self.printout(self._pages.tostring(numbers))

    def do_info(self, args):
        """
        Take one argument, page numbers (optional).

        [PyMuPDF v1.18.7 or later is required]

        Show some PDF information for *specified pages*.

        * ``Page Count`` and ``PageLabels``

        * ``MediaBox``, ``CropBox``, ``BleedBox``, ``TrimBox``, ``ArtBox``,
          ``Rotate`` and ``UserUnit``

        For boxes,
        the (almost) same values from the previous boxes are omitted.

        ``PageLabels`` and ``UserUnit`` are omitted if they are not defined.

        The values are as when PDF file was first loaded.
        User crop commands don't update them.

        Options (optional):

        ``-p``, ``--pdf``:
            print raw PDF string values as is.
            In this case, page attribute inheritances are not followed
            (``MediaBox``, ``CropBox`` and ``Rotate``).
        """
        numbers, opts = self.cmdparser.parse(args, allow_blank=True)
        if not numbers:
            return

        if opts:
            opt = opts[0]
            if opt in ('-p', '--pdf'):
                opt = 'pdf'
            else:
                opt = None
        else:
            opt = None

        self._doc.backend.print_info(numbers, printout=self.printout, opt=opt)

    def do_undo(self, args):
        """
        Take no argument.

        Undo box operations.
        """
        try:
            msg = self._doc.pages.undo()
        except UndoError as e:
            msg = e.tostring()
        self._printout(msg)

    def do_redo(self, args):
        """
        Take no argument.

        Redo box operations.
        """
        try:
            msg = self._doc.pages.redo()
        except RedoError as e:
            msg = e.tostring()
        self._printout(msg)

    def do_Set(self, args):
        """
        Take zero or two arguments, config option name and option value.

        With no argument, show current config options.

        With two arguments, set config options

        .. code-block:: none

            Set winpos 0,0
        """
        if not args:
            self._doc.print_conf_options()
        else:
            key, val = args.split(' ', maxsplit=1)
            try:
                self._doc.set_conf_option(key, val)
            except ValueError:
                pass

    def do_Python(self, args):
        """
        Take no argument.

        Run Python interpreter,
        with two variables exposed: ``doc`` and ``pages``
        (current ``Document`` and ``Document.pages`` object).

        You are supposed to know the source code.

        For now, you can use it only for reading (not writing),
        otherwise, it will terribly break undo and redo.

        (But if you are careful, not using undo and redo,
        then you *may* be able to save PDF file successfully).

        To exit *this* Python interpreter,
        run ``exit()`` or send ``EOF``.
        """
        banner = ('Entering Python interpreter...\n'
            'To exit, use exit() or send EOF')
        exitmsg = 'Exiting Python interpreter...'

        # normal 'exit()' ends the program (pdfslash) altogether.
        # (site.setquit -> _sitebuiltins.Quitter -> sys.stdin.close())
        # https://bugs.python.org/issue34115
        def exit():
            print(exitmsg)
            raise SystemExit()

        d = {
            'doc': self._doc,
            'pages': self._pages,
            'exit': exit,
            'quit': exit,
        }

        self._end_readline(self.hisfile)
        self._start_readline(self.pyhisfile)

        try:
            code.interact(banner=banner, exitmsg=exitmsg, local=d)
        except SystemExit:
            pass

        self._end_readline(self.pyhisfile)
        self._start_readline(self.hisfile)

    def do_export(self, args):
        """
        Take no argument.

        Print all box edit history in chronological order.

        Conceptually, if they are supplied as input again,
        the program should 'replay' the same edits.

        .. code-block:: none

            (pdfslash) export | cat > log.txt
            (pdfslash) exit
            $ pdfslash -f log.txt some.pdf

        It also prints file hash (crc32 to be exact) as comment,
        and 'replay' will fail if the hash is different
        from the current input PDF file ('some.pdf' in the example above).
        You can disable this check with commandline option ``--nocheck``.

        Options (optional):

        ``-a``, ``--auto``:
            write to a file automatically with the form of
            '<user directory>/exported/<PDF file name>.<timestamp>.txt'
        """
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        fname = self._doc.fname
        checksum = get_checksum(fname)
        msgs = []
        msgs.append('# %s' % t)
        msgs.append('# %s' % fname)
        msgs.append('# hash: %s' % checksum)
        stacker = self._doc.pages.boxdata.stacker
        msgs.extend(stacker.export())
        msgs = '\n'.join(msgs)

        if args and args in ('-a', '--auto'):
            fname = self._get_export_file()
            if fname:
                with open(fname, 'w') as f:
                    f.write(msgs)
        else:
            self.printout(msgs)

    def do_free(self, args):
        """
        Take no argument.

        Free all image cache in the program.
        Use when the program is grabbing too much memory.

        (The program caches almost all GUI image
        and intermediate numpy arrays).
        """
        self._doc.free()

    def do__mediabox_reset(self, args):
        """
        Take one or two arguments, page numbers (optional) and tolerance.

        (advanced, and experimental)

        Some PDF has too many slightly different mediaboxes,
        for this program to be useful (unable to group pages to preview).

        One way to solve is to choose some bigger mediabox,
        and align others to it, while discarding too different ones.
        From the program's design,
        It has to actually set new MediaBox to pages.
        This is a very crude procedure.

        PyMuPDF removes all other boxes, CropBox, BleedBox etc. (so it says).

        The process basically loads a completely new PDF,
        and resets the whole program (undo etc.), without exiting interpreter.
        Something may be broken somewhere.

        The output of 'info' command will change accordingly.

        ---

        Without optinal argument, it just reports a candidate mediabox.

        It calculates to include maximum pages in some expanded mediabox,
        excluding bigger or smaller than the tolerance given (a pixel number).

        Options (optional):

        ``-s``, ``--set``:
            actually set MediaBox to included pages,
            after reporting, using reported data.

            do nothing to excluded pages.

        Example:

        .. code-block:: none

            (pdfslash) _mediabox_reset 20           # Report for all pages,
                                                    # tolerance: 20 pixel.
            (pdfslash) _mediabox_reset 10-400 20    # Report for 10-400 pages.
            (pdfslash) _mediabox_reset 20 --set     # Set MediaBox
                                                    # for all included pages.
        """
        ret, opts = self.cmdparser.parse(args, signature='nplus')
        numbers, tolerance = ret
        tolerance = int(tolerance)
        if tolerance < 1:
            self.printout('invalid tolerance: %d' % tolerance)
            return

        setter = _PyMuPDFMediaBoxSetter(
            self._pages.mediaboxes, numbers, tolerance)
        setter.get_data()
        setter.report(printout=self.printout)

        if not opts or len(opts) != 1 or opts[0] not in ('-s', '--set'):
            return

        # create new PDF with new Mediaboxes
        pdf = setter.set_mediabox(self._doc.backend)

        self._doc.free()
        fname, conf = self._doc.fname, self._doc.conf
        doc = Document(fname, conf, pdf_obj=pdf)
        self._doc = doc

    def do_exit(self, args):
        """
        Take no argument.

        Exit the program.
        """
        self._printout('Exiting...')
        return True

    def emptyline(self):
        pass

    do_crop = do_append

    do_quit = do_exit
    do_EOF = do_exit


class _PyMuPDFMediaBoxSetter(object):
    """Manage PyMuPDF Mediabox modifications.

    Only used for PDFSlashCmd.do__mediabox_reset (exceptional case).
    """

    def __init__(self, mediaboxes, numbers, tolerance):
        self.mediaboxes = mediaboxes
        self.numbers = numbers
        self.tolerance = tolerance

    def get_data(self):
        boxes, indices = [], []
        for num, box in enumerate(self.mediaboxes, start=1):
            if num in self.numbers:
                boxes.append(box)
                indices.append(num)
        in_, out, w_high, h_high = self.get_max_box(boxes, self.tolerance)

        in_ = [indices[i] for i in in_]
        out = [indices[i] for i in out]

        self.boxes, self.in_, self.out = boxes, in_, out
        self.w, self.h = w_high, h_high

    def get_max_box(self, boxes, tolerance):
        sizes = [getsize(box) for box in boxes]

        w = sorted((s[0] for s in sizes))
        w_low, w_high, w_out = self.discard_outliers(w, tolerance)
        h = sorted((s[1] for s in sizes))
        h_low, h_high, h_out = self.discard_outliers(h, tolerance)

        in_, out = set(), set()
        for i, size in enumerate(sizes):
            if size[0] in w_out or size[1] in h_out:
                out.add(i)
            else:
                in_.add(i)

        in_ = sorted(in_)
        out = sorted(out)

        return in_, out, w_high, h_high

    def discard_outliers(self, nums, tolerance):
        out = []
        n = len(nums)
        low_index = 0
        high_index = n - 1

        while True:
            low = nums[low_index]
            high = nums[high_index]

            if high - low < tolerance:
                break

            range_ = nums[low_index:high_index + 1]
            median = range_[n // 2]  # median_high
            is_low = abs(median - low) > abs(median - high)

            if is_low:
                out.append(low)
                low_index += 1
            else:
                out.append(high)
                high_index -= 1

        return low, high, out

    def report(self, printout=print):
        nstr = g_numparser.unparse(self.in_)
        printout('expanded MediaBox size: 0,0,%d,%d' % (self.w, self.h))
        printout('included pages: %s' % nstr)
        printout('----')
        printout('excluded pages:')
        for n in self.out:
            box = self.boxes[n - 1]
            printout('%d: %d,%d,%d,%d' % (n, *box))

    def set_mediabox(self, backend):
        pdf = backend.load_pdf()
        setter = backend._compat('set_mediabox', 'setMediaBox')
        mbox = backend._compat('mediabox', 'MediaBox')

        # TODO: need centering (minus value adjustment)?
        for n in self.in_:
            page = pdf[n - 1]
            box = self.get_new_mediabox(
                self.w, self.h, mbox(page), page.rotation)
            setter(page)(box)

        return pdf

    def get_new_mediabox(self, w, h, box, rotation):
        x0, y0 = box[:2]
        x0, y0 = math.ceil(x0), math.ceil(y0)

        if rotation in (90, 270):
            box = (x0, y0, x0 + h, y0 + w)
        else:  # 0 or 180 or others
            box = (x0, y0, x0 + w, y0 + h)
        return box


class Conf(object):
    """Create config dict."""

    _env_var_dir = _ENV_VAR_DIR
    _config_filename = _CONFIG_FILENAME
    _configfunc = _CONFIGFUNC
    _conf = _CONF
    _to_skip_print = []

    def create_default(self):
        d = {}
        for k, v in self._conf.items():
            d[k] = v[0]

        d['_config_dir'] = None
        d['_self'] = self
        self._to_skip_print.append('_self')
        return d

    def create(self):
        d = self.create_default()

        config_dir = self._get_config_dir()  # sometimes it is ``None``
        if config_dir:
            user_config = self._get_user_config(config_dir)
            if user_config:
                for key, (val, funcname) in self._conf.items():
                    v = user_config.get(key)
                    if v:
                        d[key] = self._configfunc[funcname](v)

        d['_config_dir'] = config_dir
        return d

    def _get_config_dir(self):
        d = os.getenv(self._env_var_dir)
        if not d:
            config_home = os.getenv('XDG_CONFIG_HOME', '~/.config')
            d = os.path.join(config_home, 'pdfslash')
        d = os.path.expanduser(d)
        if os.path.isdir(d):
            return d

    def _get_user_config(self, config_dir):
        fname = os.path.join(config_dir, self._config_filename)
        if not os.path.isfile(fname):
            return
        config = configparser.ConfigParser()
        config.read(fname)
        if config.has_section('main'):
            return config['main']

    def set_item(self, d, key, value):
        if key not in self._conf:
            print('invalid config option: %r' % key)
            return

        func = self._configfunc[self._conf[key][1]]
        try:
            d[key] = func(value)
        except Exception:  # TODO:
            print('invalid config value: %r' % value)

    def print_items(self, d):
        for k, v in sorted(d.items()):
            if k in self._to_skip_print:
                continue
            print('%r: %r' % (k, v))


class Runner(object):
    """Run main process."""

    def __init__(self, args):
        self.args = args
        _time('start')
        self.conf = self.get_conf()
        _time('conf init')
        _time('start')
        self.doc = self.get_doc()
        _time('doc init (fitz.open)')
        _time('start')
        self.pcmd = self.get_pcmd()
        _time('Interpreter init')
        self.queue_commands()

    def get_conf(self):
        return Conf().create()

    def get_doc(self):
        return Document(self.args.pdffile, self.conf)

    def get_pcmd(self):
        return PDFSlashCmd(doc=self.doc)

    def queue_commands(self):
        if self.args.cmdfile:
            try:
                with open(self.args.cmdfile) as f:
                    commands = f.read()
            except (FileNotFoundError, IsADirectoryError):
                fmt = 'Error: file not found, or is directory: %r'
                print(fmt % self.args.cmdfile)
                sys.exit(1)
        elif self.args.command:
            commands = self.args.command
        else:
            return

        cmdqueue = []
        for command in commands.split('\n'):
            for c in command.split(';'):
                c = self.parse_command(c)
                if c is not None:
                    cmdqueue.append(c)

        self.pcmd.cmdqueue = cmdqueue

    def parse_command(self, line):
        line = line.strip()
        if not line:
            return
        if line.startswith('#'):
            self.check_pdffile(line)
            return

        m = re.match(r'\[[a-z]+\] ', line)
        if m:
            return line[m.end():]
        return line

    def check_pdffile(self, line):
        intro = '# hash: '
        if line.startswith(intro) and not self.args.nocheck:
            oldhash = line[len(intro):]
            current = get_checksum(self.args.pdffile)
            if oldhash != current:
                fmt = ('Different file hash from one in initial commands: '
                    'old: %s, current: %s.')
                raise ValueError(fmt % (oldhash, current))

    def run(self):
        if self.args._nobanner:
            intro = ''
        else:
            intro = None
        return self.pcmd.cmdloop(intro=intro)


class DefaultRunner(Runner):
    """Run main process with default config (for test)."""

    def get_conf(self):
        return Conf().create_default()


def _build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    h = 'PDF filename to process'
    parser.add_argument('pdffile', metavar='PDFFILE', help=h)

    h = ('run initial commands before showing prompt '
        "(split multiple commands with ';').")
    parser.add_argument('--command', '-c', help=h)

    h = ('run initial commands before showing prompt '
        "(reading from a file, one command a line).")
    parser.add_argument('--cmdfile', '-f', help=h)

    h = ('do not perform initial commands verification. '
        "Otherwise the program aborts when a line starts with '# hash: ', "
        'and the value is different from input PDF file. '
        "This is for 'export' command.")
    parser.add_argument('--nocheck', '-n', action='store_true', help=h)

    h = '[DEBUG] print time for some processes'
    parser.add_argument('--_time', '-_t', action='store_true', help=h)

    h = '[DEBUG] save merged image used for GUI in current directory'
    parser.add_argument('--_save', '-_s', action='store_true', help=h)

    h = '[DEBUG] suspend interpreter banner (intro)'
    parser.add_argument('--_nobanner', '-_b', action='store_true', help=h)

    return parser


def main(args=None, runner=None):
    args = args or sys.argv[1:]
    parser = _build_argument_parser()
    args = parser.parse_args(args)

    if args._time:
        global _PRINT_TIME
        _PRINT_TIME = True

    if args._save:
        global _SAVE_IMG
        _SAVE_IMG = True

    runner = runner or Runner
    runner = runner(args)
    runner.run()


if __name__ == '__main__':
    main()
