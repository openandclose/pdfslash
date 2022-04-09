
"""Test PDF sepecific operations."""

import fitz

import pdfslash.slash as slash


# Rotation ---------------------------------------

def _get_data():
    mediabox = (0, 0, 9, 14)
    boxes = (
        ((1, 2, 6, 10), 0),
        ((4, 1, 12, 6), 90),
        ((3, 4, 8, 12), 180),
        ((2, 3, 10, 8), 270),
    )
    w, h = mediabox[2:]
    cbox = boxes[0][0]
    return w, h, cbox, boxes


def test_rotation():
    w, h, cbox, boxes = _get_data()
    for box, rot in boxes:
        assert slash.rotate(w, h, rot, cbox) == box


def test_unrotation():
    w, h, cbox, boxes = _get_data()
    for box, rot in boxes:
        assert slash.unrotate(w, h, rot, box) == cbox


def _test_fitz_unrotation():
    # check returning the same results as test_unrotation.
    def get_fitz_unrotated(page, box):
        box = fitz.Rect(box)
        box = box * page.derotation_matrix
        box = box.normalize()
        return slash.ints(box)

    w, h, cbox, boxes = _get_data()
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    for box, rot in boxes:
        page.set_rotation(rot)
        assert get_fitz_unrotated(page, box) == cbox


# ImgBox ----------------------------------------

def isclose(box1, box2):
    for a, b in zip(box1, box2):
        if abs(a - b) > 0.0001:
            return False
    return True


def test_cropbox_pos():
    def check(mbox, new_cropbox, expected1, expected2):
        doc = fitz.open()
        w, h = mbox[2] - mbox[0], mbox[3] - mbox[1]
        page = doc.new_page(width=w, height=h)
        cbox = page.cropbox
        b = slash._PyMuPDFImgBox(mbox, cbox, 0)
        new_cbox = b.cropbox2cbox(new_cropbox)
        assert isclose(new_cbox, expected1) is True
        page.set_cropbox(new_cbox)
        assert doc.xref_get_key(page.xref, 'CropBox')[1] == expected2

    mbox = 0, 0, 9, 14
    new_cropbox = 1, 2, 6, 10
    expected1 = 1, 2, 6, 10
    expected2 = '[1 4 6 12]'
    check(mbox, new_cropbox, expected1, expected2)

    mbox = 0, 0, 9.1, 14.1
    new_cropbox = 1, 2, 6, 10
    expected1 = 1, 2.1, 6, 10.1
    expected2 = '[1 4 6 12]'
    check(mbox, new_cropbox, expected1, expected2)

    mbox = 0.1, 0.1, 10, 15
    new_cropbox = 1, 2, 6, 10
    expected1 = 1.1, 2.9, 6.1, 10.9
    expected2 = '[1.1 4 6.1 12]'
    check(mbox, new_cropbox, expected1, expected2)

    mbox = 0.1, 0.1, 10.1, 15.1
    new_cropbox = 1, 2, 6, 10
    expected1 = 1.1, 2, 6.1, 10
    expected2 = '[1.1 5 6.1 13]'
    check(mbox, new_cropbox, expected1, expected2)

    mbox = 0.1, 0.1, 10.2, 15.2
    new_cropbox = 1, 2, 6, 10
    expected1 = 1.1, 2.1, 6.1, 10.1
    expected2 = '[1.1 5 6.1 13]'
    check(mbox, new_cropbox, expected1, expected2)

    mbox = -0.1, -0.1, 10, 15
    new_cropbox = 1, 2, 6, 10
    expected1 = 0.9, 2.1, 5.9, 10.1
    expected2 = '[.9 5 5.9 13]'
    check(mbox, new_cropbox, expected1, expected2)


# Page Labels ------------------------------------

def test_labels():
    doc = fitz.open()
    for i in range(20):
        doc.new_page()

    labels = [
        {'startpage': 0, 'prefix': 'A-', 'style': 'D', 'firstpagenum': 1},
        {'startpage': 4, 'prefix': '', 'style': 'R', 'firstpagenum': 1},
        {'startpage': 11, 'prefix': '', 'style': 'D', 'firstpagenum': 1},
    ]
    doc.set_page_labels(labels)

    backend = slash.PyMuPDFBackend(fname='', pdf_obj=doc)
    # Must instansiate slash._Pages class to fill global slash.g_numparser
    slash._Pages(backend.mediaboxes, backend.cropboxes)

    p = backend._format_labels

    numbers = tuple(range(1, 21))
    expected = 'A-1, A-2, A-3, A-4, I, II, III, IV, V, VI, VII, 1-9'
    assert p(numbers) == expected

    numbers = (1,)
    expected = 'A-1'
    assert p(numbers) == expected

    numbers = (20,)
    expected = '9'
    assert p(numbers) == expected

    numbers = tuple(range(4, 9))
    expected = 'A-4, I, II, III, IV'
    assert p(numbers) == expected

    numbers = tuple(range(9, 14))
    expected = 'V, VI, VII, 1,2'
    assert p(numbers) == expected
