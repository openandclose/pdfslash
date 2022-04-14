
"""Test base data operations."""

import pdfslash.slash as slash

import pytest


class NameSpace(object):
    pass


class TestStacker:

    def create_data(self):
        return [[1, 1, 1] for _ in range(4)]

    def check(self, stacker, index, expected):
        assert stacker._data[index] == expected

    def test_op(self):
        stacker = slash._Stacker(self.create_data())

        stacker.set([('add', (0, 0), 55)])
        self.check(stacker, 0, [55, 1, 1, 1])
        stacker.set([('replace', (0, 0), 66)])
        self.check(stacker, 0, [66, 1, 1, 1])
        stacker.set([('remove', (0, 0), None)])
        self.check(stacker, 0, [1, 1, 1])

        stacker = slash._Stacker(self.create_data())

        stacker.set([('add', (1, 1), 55)], 'adding')
        self.check(stacker, 1, [1, 55, 1, 1])
        stacker.set([('replace', (1, 1), 66)], 'replacing')
        self.check(stacker, 1, [1, 66, 1, 1])
        stacker.set([('remove', (1, 1), None)], 'removing')
        self.check(stacker, 1, [1, 1, 1])

        assert stacker.undo() == 'removing'
        self.check(stacker, 1, [1, 66, 1, 1])
        assert stacker.undo() == 'replacing'
        self.check(stacker, 1, [1, 55, 1, 1])
        assert stacker.undo() == 'adding'
        self.check(stacker, 1, [1, 1, 1])
        assert stacker._data == self.create_data()
        with pytest.raises(slash.UndoError):
            stacker.undo()

        assert stacker.redo() == 'adding'
        self.check(stacker, 1, [1, 55, 1, 1])
        assert stacker.redo() == 'replacing'
        self.check(stacker, 1, [1, 66, 1, 1])
        assert stacker.redo() == 'removing'
        self.check(stacker, 1, [1, 1, 1])
        assert stacker._data == self.create_data()
        with pytest.raises(slash.RedoError):
            stacker.redo()


class TestBoxData:

    def check(self, boxdata, numbers, expected):
        for n in numbers:
            assert repr(boxdata.boxes[n - 1]) == str(expected)

    def check_rects(self, boxdata, data):
        for box, numbers in data:
            assert boxdata.boxdict.rects[box] == numbers

    def test_command(self, capsys):
        box1 = (100, 100, 300, 300)
        box2 = (200, 200, 500, 500)

        numbers = tuple(range(1, 9))
        boxdata = slash._BoxData(numbers)

        # 1
        boxdata.overwrite((1, 2, 3), box1, msg='msg1')
        self.check(boxdata, (1, 2, 3), [box1])
        self.check_rects(boxdata, [(box1, [1, 2, 3])])

        # 2
        boxdata.overwrite((1, 2, 3), box2, msg='msg2')
        self.check(boxdata, (1, 2, 3), [box2])
        self.check_rects(boxdata, [(box2, [1, 2, 3])])

        # undo: back to 1
        assert boxdata.undo() == 'msg2'
        self.check(boxdata, (1, 2, 3), [box1])
        self.check_rects(boxdata, [(box1, [1, 2, 3])])

        # 2
        boxdata.append((2, 3, 4), box2, msg='msg2')
        self.check(boxdata, (1,), [box1])
        self.check(boxdata, (2, 3), [box1, box2])
        self.check(boxdata, (4,), [box2])
        self.check_rects(boxdata, [(box1, [1, 2, 3]), (box2, [2, 3, 4])])

        # 3
        boxdata.clear((2, 3), msg='msg3')
        self.check(boxdata, (2, 3), [])
        self.check_rects(boxdata, [(box1, [1]), (box2, [4])])

        # error (can't discard non-existent box)
        boxdata.discard((1,), box2, msg='error!')
        self.check(boxdata, (1,), [box1])
        self.check_rects(boxdata, [(box1, [1])])
        assert 'NoBoxToProcessError: ' in capsys.readouterr().out.rstrip()

        # 4
        boxdata.discard((1,), box1, msg='msg4')
        self.check(boxdata, (1,), [])
        self.check_rects(boxdata, [(box1, [])])  # leaving blank list

        # error (can't append a box already there)
        boxdata.append((4,), box2, msg='error!')
        self.check(boxdata, (4,), [box2])
        self.check_rects(boxdata, [(box2, [4])])
        assert 'DuplicateBoxError: ' in capsys.readouterr().out.rstrip()

        # 5 (for modify)
        boxdata.modify((4,), box1, box2, msg='msg5')
        self.check(boxdata, (4,), [box1])
        self.check_rects(boxdata, [(box1, [4]), (box2, [])])

        # 6 (for modify)
        boxdata.modify((4,), box2, box1, msg='msg6')
        self.check(boxdata, (4,), [box2])
        self.check_rects(boxdata, [(box1, []), (box2, [4])])

        # 7 (for set_each)
        commands = [
            ('append', 1, box1),
            ('overwrite', 2, box2),
            ('modify', 4, box1, box2),
        ]
        boxdata.set_each(commands, msg='msg7')
        self.check(boxdata, (1,), [box1])
        self.check(boxdata, (2,), [box2])
        self.check(boxdata, (4,), [box1])
        self.check_rects(boxdata, [(box1, [1, 4]), (box2, [2])])

        assert boxdata.undo() == 'msg7'
        assert boxdata.undo() == 'msg6'
        assert boxdata.undo() == 'msg5'
        assert boxdata.undo() == 'msg4'
        assert boxdata.undo() == 'msg3'
        assert boxdata.undo() == 'msg2'
        assert boxdata.undo() == 'msg1'
        assert [b.data for b in boxdata.boxes] == [[] for _ in range(8)]


class TestNumParser:

    def test_parse(self):
        p = slash.NumParser(length=100).parse

        assert p('') == []

        assert p('1') == (1,)
        assert p('1,2,3,4') == (1, 2, 3, 4)
        assert p('1,3,5,7') == (1, 3, 5, 7)
        assert p('8-9') == (8, 9)
        assert p('8-12') == (8, 9, 10, 11, 12)
        assert p('11^13') == (11, 13)
        assert p('11^17') == (11, 13, 15, 17)
        assert p('14^18') == (14, 16, 18)
        assert p('3,6,8-10,33') == (3, 6, 8, 9, 10, 33)
        assert p('98-') == (98, 99, 100)

        assert p('~') == (98, 99, 100)
        assert p(':') == list(range(1, 101))

        assert p('98-103') == (98, 99, 100)
        assert p('101-103') == ()

        with pytest.raises(ValueError):
            assert p('a') == 0
        with pytest.raises(ValueError):
            assert p('--') == 0
        with pytest.raises(ValueError):
            assert p('5--') == 0
        with pytest.raises(ValueError):
            assert p('-5-') == 0
        with pytest.raises(ValueError):
            assert p('--5') == 0

        with pytest.raises(ValueError):
            assert p('0') == 0
        with pytest.raises(ValueError):
            assert p('7-6') == 0
        with pytest.raises(ValueError):
            assert p('7^10') == 0
        with pytest.raises(ValueError):
            assert p('3-5,-,10-12') == 0
        with pytest.raises(ValueError):
            assert p('3-5,7-,10-12') == 0

    def test_unparse(self):
        u = slash.NumParser(length=100).unparse

        assert u([1]) == '1'
        assert u([5, 8, 11]) == '5,8,11'
        assert u([6, 7, 11]) == '6,7,11'
        assert u([6, 8, 11]) == '6,8,11'
        assert u([6, 7, 8, 11]) == '6-8,11'
        assert u([6, 8, 10, 11]) == '6^10,11'
        assert u([6, 8, 9, 10, 12]) == '6,8-10,12'
        assert u([6, 7, 9, 11, 12]) == '6,7^11,12'
        assert u([6, 8, 9, 11, 12]) == '6,8,9,11,12'


class TestCommandParser:

    def create(self):
        def create_pages():
            boxes = [(0, 0, 595, 842) for _ in range(8)]
            return slash._Pages(boxes, boxes)

        def create_cmd(pages):
            cmd = NameSpace()
            cmd.numparser = slash.NumParser(len(pages))
            cmd.boxparser = slash.BoxParser(pages)
            cmd.printout = print
            return cmd

        pages = create_pages()
        cmd = create_cmd(pages)
        return slash.CommandParser(cmd), pages

    def test_parse(self):
        parser, pages = self.create()

        b1, B1 = '100,110,120,130', (100, 110, 120, 130)
        b2, B2 = '200,210,220,230', (200, 210, 220, 230)
        b3, B3 = '300,310,320,330', (300, 310, 320, 330)
        b32, B32 = '340,350,360,370', (340, 350, 360, 370)
        b4, B4 = '400,410,420,430', (400, 410, 420, 430)
        x5, X5 = '500,510,520,530', (500, 510, 520, 530)

        pages.overwrite((1,), B1)
        pages.overwrite((2,), B2)
        pages.overwrite((3,), B3)
        pages.append((3,), B32)
        pages.append((3,), B1)
        pages.overwrite((4,), B4)

        def p(*s):
            return parser.parse(' '.join(s[:-1]), s[-1])[0]

    # numbers only
        assert p('1', 'n') == (1,)

    # append or overwrite (numbers, box)
        assert p('1', x5, 'nb') == ((1,), X5)
        assert p('1-3', x5, 'nb') == ((1, 2, 3), X5)

    # discard (numbers, box)
        assert p('3', b1, 'nB') == ((3,), B1)
        assert p('1,3', b1, 'nB') == ((1, 3), B1)
        assert p('2', b1, 'nB') == ((2,), B1)  # no error (no printout)
        # pages.discard((2), B1)  # no error (printout)

    # modify (numbers, box1, box2)
        assert p('1', b1, x5, 'nbb') == ('modify', (1,), B1, X5)
        assert p('1,3', b1, x5, 'nbb') == ('modify', (1, 3), B1, X5)
        assert p('1', '@1', x5, 'nbb') == ('modify', (1,), B1, X5)
        assert p('3', '@1', x5, 'nbb') == ('modify', (3,), B3, X5)
        assert p('3', '@3', x5, 'nbb') == ('modify', (3,), B1, X5)
        assert p('1,3', '@1', x5, 'nbb') == (
                'set_each', [('modify', 1, B1, X5), ('modify', 3, B3, X5)])

        mo = 'modify'
        bstr2 = '-100,-100,+100,+100'
        box2_1 = 0, 10, 220, 230  # box2 result of page 1
        box2_3 = 200, 210, 420, 430  # box2 result of page 3
        assert p('1', b1, bstr2, 'nbb') == ('modify', (1,), B1, box2_1)
        assert p('1', '@1', bstr2, 'nbb') == ('modify', (1,), B1, box2_1)
        assert p('1,3', '@1', bstr2, 'nbb') == (
                'set_each', [(mo, 1, B1, box2_1), (mo, 3, B3, box2_3)])

        bstr2 = 'min,min,max,max'
        box2_m = 100, 110, 320, 330
        assert p('1,3', '@1', bstr2, 'nbb') == (
                'set_each', [(mo, 1, B1, box2_m), (mo, 3, B3, box2_m)])
