
"""Test PDF sepecific operations."""

import fitz

import pdfslash.slash as slash


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
