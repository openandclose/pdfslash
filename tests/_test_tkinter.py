
"""Test tkinter.

I'm following ivan_pozdeev's great answer for how to test tkinter.
https://stackoverflow.com/a/49028688
"""

import _tkinter
import os

import pdfslash.slash as slash

COLORS = slash.COLORS

# import pytest


# Created from editing PDF/UA Reference Suite 1.1
# PDFUA-Ref-2-05_BookChapter-german.pdf
# https://www.pdfa.org/resource/pdfua-reference-suite/
# It has 3 different source cropboxes:
#
# page          cropbox             size
# 1-13,21       0,0,595,841         595,841
# 14-16         30,30,560,810       530,780    (by 'crop 14-16 30,30,560,810')
# 17-20         50,50,550,800       500,750    (by 'crop 17-20 50,50,550,800')
PDFF = 'PDFUA-Ref-2-05_BookChapter-german-3cboxes.pdf'
PDFF = os.path.join(os.path.dirname(__file__), PDFF)


# import time
# def teardown_function(function):
#     time.sleep(5)


def _print_rects(t):
    print()
    print('    tkinter:')
    for r in t.canvas.find_withtag('rect'):
        state = t.canvas.itemcget(r, 'state') or '      '
        tags = t.canvas.itemcget(r, 'tags')
        box = t.canvas.bbox(r)
        color = t.canvas.itemcget(r, 'outline')
        for k, v in COLORS.items():
            if v == color:
                colorname = k
        fmt = '    gid: %s, state: %s, tags: %s, box: %s, color: %s'
        print(fmt % (r, state, tags, box, colorname))
    print()
    print('    data:')
    for r in t.i.rects.rects.values():
        fmt = '    _box: %s, _tempbox: %s, gid: %s, active: %s, num: %s'
        nstr = t.i._numparser.unparse(r.numbers)
        print(fmt % (r._box, r._tempbox, r.gid, r.active, nstr))
    print()
    print('    boxdict:')
    rects = t._doc.pages.boxdata.boxdict.rects
    for k, v in rects.items():
        print('    %r: %r' % (k, v))


class Event(object):
    """Simulate tkinter event class."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def pump_events(root):
    while root.dooneevent(_tkinter.ALL_EVENTS | _tkinter.DONT_WAIT):
        pass


def build(numbers=None):
    parser = slash._build_argument_parser()
    args = parser.parse_args([PDFF])
    runner = slash.DefaultRunner(args)
    return runner.doc


def run(doc, numbers=None):
    numbers = numbers or (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    tkrunner = doc._get_tkrunner(numbers)
    return tkrunner


def close(t):
    t.quit(Event())


def create_sel(t, box):
    x0, y0, x1, y1 = box

    event = Event(x=x0, y=y0)
    t._set_start(event)
    pump_events(t.root)

    event = Event(x=x1, y=y1)
    t._set_end(event)
    pump_events(t.root)


def press_key(t, key, times=1):
    for _ in range(times):
        t.root.event_generate(key)
        pump_events(t.root)


def check_tk_boxes(t, numbers, boxes):
    for number in numbers:
        assert t._doc.pages[number].boxes.data == boxes


def check_tk_rects(t, data):  # data is a list of rect.box, rect.color
    rects = [r for r in t.canvas.find_withtag('rect')
        if t.canvas.itemcget(r, 'state') in ('', 'normal')]
    assert len(rects) == len(data)

    rects = sorted(rects, key=lambda r: t.canvas.bbox(r))
    for r, (box, color) in zip(rects, data):
        assert COLORS[color] == t.canvas.itemcget(r, 'outline')
        is_same_box(box, t.canvas.bbox(r))


def check_tk_image(t, m_shape, c_shape):
    width = t.i.width
    height = t.i.height
    is_same_box(m_shape, (width, height))

    gid = t.canvas.find_withtag('cropbox')[0]
    if c_shape is None:
        return
    is_same_box(c_shape, t.canvas.bbox(gid))


def is_same_box(box, rect):
    # It seems outline width (1) is added externally to the shape.
    box = slash.shift_box(box, (-1, -1, 1, 1))
    for b, r in zip(box, rect):
        # assert abs(b - r) < 2
        b == r


def move_rect(t, which='rightdown', times=10):  # rightdown or leftup
    if which == 'rightdown':
        keys = '<Right>', '<Down>', '<Shift-Right>', '<Shift-Down>'
    else:
        keys = '<Left>', '<Up>', '<Shift-Left>', '<Shift-Up>'

    for key in keys:
        press_key(t, key, times)


def get_scale(t):
    return t.i._scaling.scale


def get_scaled(t, coords):
    return t.i._scaling.get_scaled(coords)


def test_rects():
    numbers = tuple(i for i in range(1, 22))
    doc = build()
    t = run(doc, numbers)
    pump_events(t.root)

    # create
    box1 = 100, 120, 200, 220
    create_sel(t, box1)
    press_key(t, '<Return>')
    check_tk_boxes(t, (1, 2, 3), [box1])
    check_tk_rects(t, ((box1, 'blue'),))

    # append
    box2 = 300, 320, 500, 520
    create_sel(t, box2)
    press_key(t, '<Return>')
    check_tk_boxes(t, (1, 2, 3), [box1, box2])
    check_tk_rects(t, ((box1, 'blue'), (box2, 'blue')))

    # (undo)
    press_key(t, '<u>')
    check_tk_boxes(t, (1, 2, 3), [box1])
    check_tk_rects(t, ((box1, 'blue'),))

    # overwrite
    box3 = 105, 126, 207, 228
    create_sel(t, box3)
    press_key(t, '<Shift-Return>')
    check_tk_boxes(t, (1, 2, 3), [box3])
    check_tk_rects(t, ((box3, 'blue'),))

    # odd
    press_key(t, '<v>')  # odds
    box4 = 310, 330, 510, 530
    create_sel(t, box4)
    press_key(t, '<Return>')
    check_tk_boxes(t, (1, 3), [box3, box4])
    check_tk_boxes(t, (2, 4), [box3])
    check_tk_rects(t, ((box3, 'blue'), (box4, 'blue')))
    press_key(t, '<v>')  # evens
    check_tk_rects(t, ((box3, 'blue'),))
    press_key(t, '<v>')  # all
    check_tk_rects(t, ((box3, 'blue'), (box4, 'lightblue')))

    # active
    press_key(t, '<a>')  # box3
    check_tk_rects(t, ((box3, 'orange'), (box4, 'lightblue')))
    press_key(t, '<a>')  # box4
    check_tk_rects(t, ((box3, 'blue'), (box4, 'orange')))
    press_key(t, '<a>')  # sel
    check_tk_rects(t, ((box3, 'blue'), (box4, 'lightblue')))

    # modify
    press_key(t, '<a>')  # box3
    move_rect(t, 'rightdown')
    press_key(t, '<Return>')
    box3a = tuple(b + 10 for b in box3)
    check_tk_boxes(t, (1, 3), [box3a, box4])
    check_tk_boxes(t, (2, 4), [box3a])
    check_tk_rects(t, ((box3a, 'blue'), (box4, 'lightblue')))

    press_key(t, '<a>')  # box3
    move_rect(t, 'leftup')
    press_key(t, '<Return>')
    check_tk_boxes(t, (1, 3), [box3, box4])
    check_tk_boxes(t, (2, 4), [box3])
    check_tk_rects(t, ((box3, 'blue'), (box4, 'lightblue')))

    press_key(t, '<a>')  # box3
    move_rect(t, 'leftup')
    press_key(t, '<Return>')
    box3b = tuple(b - 10 for b in box3)
    check_tk_boxes(t, (1, 3), [box3b, box4])
    check_tk_boxes(t, (2, 4), [box3b])
    check_tk_rects(t, ((box3b, 'blue'), (box4, 'lightblue')))

    # img groups
    isize = (595, 841)  # mediabox size
    shape1 = (0, 0, 595, 841)  # crpobox coords 1-13,21
    shape2 = (30,30,560,810)  # crpobox coords 14-16
    shape3 = (50,50,550,800)  # crpobox coords 17-20

    check_tk_image(t, isize, None)
    press_key(t, '<n>')  # next
    check_tk_image(t, isize, shape2)
    press_key(t, '<n>')  # next
    check_tk_image(t, isize, shape3)
    press_key(t, '<n>')  # next
    check_tk_image(t, isize, None)
    press_key(t, '<p>')  # previous
    check_tk_image(t, isize, shape3)
    press_key(t, '<p>')  # previous
    check_tk_image(t, isize, shape2)

    box11 = 100, 100, 400, 600  # on shape2
    create_sel(t, box11)
    press_key(t, '<Return>')
    check_tk_boxes(t, (14, 15, 16), [box11])
    check_tk_rects(t, ((box11, 'blue'),))

    press_key(t, '<n>')  # next (on shape3)
    box12 = 150, 150, 450, 650
    create_sel(t, box12)
    press_key(t, '<Return>')
    check_tk_boxes(t, (14, 15, 16), [box11])
    check_tk_boxes(t, (17, 18, 19, 20), [box12])
    check_tk_rects(t, ((box12, 'blue'),))

    press_key(t, '<u>')  # undo (back to shape2)
    check_tk_image(t, isize, shape2)
    check_tk_boxes(t, (14, 15, 16), [box11])
    check_tk_boxes(t, (17, 18, 19, 20), [])
    check_tk_rects(t, ((box11, 'blue'),))

    press_key(t, '<u>')  # undo (back to isize)
    check_tk_image(t, isize, None)
    check_tk_boxes(t, (14, 15, 16), [])
    check_tk_boxes(t, (17, 18, 19, 20), [])
    check_tk_rects(t, ((box3b, 'blue'), (box4, 'lightblue')))

    press_key(t, '<r>')  # redo (back to shape2)
    check_tk_image(t, isize, shape2)
    check_tk_boxes(t, (14, 15, 16), [box11])
    check_tk_boxes(t, (17, 18, 19, 20), [])
    check_tk_rects(t, ((box11, 'blue'),))

    # zoom
    assert get_scale(t) == 1
    check_tk_image(t, isize, shape2)
    press_key(t, '<z>')
    assert get_scale(t) == 1.1
    check_tk_image(t, get_scaled(t, isize), get_scaled(t, shape2))
    box11z = get_scaled(t, box11)
    check_tk_boxes(t, (14, 15, 16), [box11])
    check_tk_rects(t, ((box11z, 'blue'),))

    press_key(t, '<Z>')
    press_key(t, '<Z>')
    assert get_scale(t) == 0.9
    check_tk_image(t, get_scaled(t, isize), get_scaled(t, shape2))
    box11ZZ = get_scaled(t, box11)
    check_tk_boxes(t, (14, 15, 16), [box11])
    check_tk_rects(t, ((box11ZZ, 'blue'),))

    close(t)


def test_rects__bug_img_group_mix_up_on_second_tk_invocation():
    numbers = tuple(i for i in range(1, 22))
    doc = build()
    t = run(doc, numbers)
    pump_events(t.root)

    press_key(t, '<q>')

    t = run(doc, numbers)
    pump_events(t.root)

    # img groups
    isize = (595, 841)  # mediabox size
    shape1 = (0, 0, 595, 841)  # crpobox coords 1-13,21
    shape2 = (30,30,560,810)  # crpobox coords 14-16
    shape3 = (50,50,550,800)  # crpobox coords 17-20

    check_tk_image(t, isize, None)
    press_key(t, '<n>')  # next
    check_tk_image(t, isize, shape2)
    press_key(t, '<n>')  # next
    check_tk_image(t, isize, shape3)
    press_key(t, '<n>')  # next
    check_tk_image(t, isize, None)

    close(t)
