
"""Test PDF sepecific operations."""

import fitz

import pdfslash.slash as slash


# Rotation ---------------------------------------

mediabox = (0, 0, 9, 14)
boxes = (
    (1, 2, 6, 10),  # rotation: 0
    (4, 1, 12, 6),  # rotation: 90
    (3, 4, 8, 12),  # rotation: 180
    (2, 3, 10, 8),  # rotation: 270
)


def get_fitz_unrotated(page, box):
    box = fitz.Rect(box)
    box = box * page.derotation_matrix
    box = box.normalize()
    return slash.ints(box)


def test_rotation():
    rot = 0
    w, h = mediabox[2:]
    doc = fitz.open()
    for _, box in zip(range(4), boxes):
        page = doc.new_page(width=w, height=h)
        page.set_rotation(rot)
        assert slash.unrotate(w, h, rot, box) == boxes[0]
        rot += 90


def _test_fitz_rotation():
    # check returning the same results as test_rotation.
    rot = 0
    w, h = mediabox[2:]
    doc = fitz.open()
    for _, box in zip(range(4), boxes):
        page = doc.new_page(width=w, height=h)
        page.set_rotation(rot)
        assert get_fitz_unrotated(page, box) == boxes[0]
        rot += 90


# Page Labels ------------------------------------

def test_labels():
    doc = fitz.open()
    for i in range(20):
        page = doc.new_page()

    labels = [
        {'startpage': 0, 'prefix': 'A-', 'style': 'D', 'firstpagenum': 1},
        {'startpage': 4, 'prefix': '', 'style': 'R', 'firstpagenum': 1},
        {'startpage': 11, 'prefix': '', 'style': 'D', 'firstpagenum': 1},
    ]
    doc.set_page_labels(labels)

    backend = slash.PyMuPDFBackend(fname='', pdf_obj=doc)
    # Must instansiate slash._Pages class to fill global slash.g_numparser
    pages = slash._Pages(backend.mediaboxes, backend.cropboxes)

    p = backend._format_labels
    ret = []
    printout = ret.append

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
