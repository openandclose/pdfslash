
"""Test base data operations."""

import pdfslash.slash as slash

import pytest


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
        assert stacker.undo() == None

        assert stacker.redo() == 'adding'
        self.check(stacker, 1, [1, 55, 1, 1])
        assert stacker.redo() == 'replacing'
        self.check(stacker, 1, [1, 66, 1, 1])
        assert stacker.redo() == 'removing'
        self.check(stacker, 1, [1, 1, 1])
        assert stacker._data == self.create_data()
        assert stacker.redo() == None


class TestBoxData:

    def create_data(self):
        return [(0, 0, 595, 842) for _ in range(8)]

    def check(self, boxdata, numbers, expected):
        for n in numbers:
            assert repr(boxdata.boxes[n - 1]) == str(expected)

    def check_rects(self, boxdata, data):
        for box, numbers in data:
            assert boxdata.boxdict.rects[box] == numbers

    def test_command(self, capsys):
        box1 = (100, 100, 300, 300)
        box2 = (200, 200, 500, 500)

        boxdata = slash._BoxData(self.create_data())

        # 1
        boxdata.overwrite((1, 2, 3), box1, msg='msg1')
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
        assert 'NoBoxToRemoveError: ' in capsys.readouterr().out.rstrip()

        # 4
        boxdata.discard((1,), box1, msg='msg4')
        self.check(boxdata, (1,), [])
        self.check_rects(boxdata, [(box1, [])])  # leaving blank list

        # error (can't append a box already there)
        boxdata.append((4,), box2, msg='error!')
        self.check(boxdata, (4,), [box2])
        self.check_rects(boxdata, [(box2, [4])])
        assert 'DuplicateBoxError: ' in capsys.readouterr().out.rstrip()

        assert boxdata.undo() == 'msg4'
        assert boxdata.undo() == 'msg3'
        assert boxdata.undo() == 'msg2'
        assert boxdata.undo() == 'msg1'
        assert [b.data for b in boxdata.boxes] == [[] for _ in range(8)]

class TestNumParser:

    def test_parse(self):
        p = slash.NumParser(length=100).parse

        assert p('1') == [1]
        assert p('8-9') == [8, 9]
        assert p('8-12') == [8, 9, 10, 11, 12]
        assert p('11^13') == [11, 13]
        assert p('11^17') == [11, 13, 15, 17]
        assert p('14^18') == [14, 16, 18]
        assert p('3,6,8-10,33') == [3, 6, 8, 9, 10, 33]
        assert p('98-') == [98, 99, 100]

        assert p('~') == [98, 99, 100]
        assert p(':') == list(range(1, 101))

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


class TestBoxParser:

    def create_data(self):
        """
        boxes = [
            [(100, 110, 120, 130)],
            [(200, 210, 220, 230)],
            [(300, 310, 320, 330), (340, 350, 360, 370)],
            [(400, 410, 420, 430)],
            [],
            [],
            [],
            [],
        ]
        """
        cboxes = [(0, 0, 595, 842) for _ in range(8)]
        pages = slash._Pages(cboxes)
        pages.overwrite((1,), (100, 110, 120, 130))
        pages.overwrite((2,), (200, 210, 220, 230))
        pages.overwrite((3,), (300, 310, 320, 330))
        pages.append((3,), (340, 350, 360, 370))
        pages.overwrite((4,), (400, 410, 420, 430))
        return pages

    def test_parse(self):
        pages = self.create_data()
        p = slash.BoxParser(pages).parse

        assert p((1,), '100,110,120,130') == (
            'crop', (1,), (100, 110, 120, 130))
        assert p((1,), '500,510,520,530') == (
            'crop', (1,), (500, 510, 520, 530))
        assert p((1, 2), '500,510,520,530') == (
            'crop', (1, 2), (500, 510, 520, 530))
        assert p((1, 2), 'min,min,max,max') == (
            'crop', (1, 2), (100, 110, 220, 230))
        assert p((1, 2, 3), 'min,min,max,max') == (
            'crop', (1, 2, 3), (100, 110, 360, 370))

        assert p((1,), '-100,-100,+400,+400') == (
            'crop_each', (1,), [[(0, 10, 520, 530)]])

        assert p((2, 3), '111,max,+200,530') == (
            'crop_each', (2, 3),
            [[(111, 350, 420, 530)],
                [(111, 350, 520, 530), (340, 350, 360, 370)]])
