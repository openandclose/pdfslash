
"""Test img (numpy 2d arrays)."""

import numpy

import pdfslash.slash as slash


def test_imgproxy():
    root_array = numpy.zeros(2000).reshape(20, 10, 10)
    root_imgs = numpy.arange(2000).reshape(20, 10, 10)

    indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    array = root_array[indices]
    loader = lambda i: root_imgs[i]
    proxy = slash._ImgProxy(array, loader, indices)

    proxy = proxy[[2, 4, 7]]
    imgs, cnt = proxy.load()
    assert list(imgs[:, 0, 0]) == [500, 900, 1500]

    proxy = proxy[[1, 2]]
    imgs, cnt = proxy.load()
    assert list(imgs[:, 0, 0]) == [900, 1500]

    proxy = proxy[[1]]
    imgs, cnt = proxy.load()
    assert list(imgs[:, 0, 0]) == [1500]

    proxy = proxy[[0]]
    imgs, cnt = proxy.load()
    assert list(imgs[:, 0, 0]) == [1500]


def test_imggroup():
    def build_imggroup():
        ma = (100, 100, 300, 300)  # mediabox A (has cropbox a and b)
        ca = (110, 110, 290, 290)  # cropbox a
        cb = (120, 120, 280, 280)  # cropbox b
        mb = (200, 200, 500, 500)  # mediabox B (has cropbox c and d)
        cc = (210, 210, 490, 490)  # cropbox c
        cd = (220, 220, 480, 480)  # cropbox d
        rep = lambda x, i: [x for _ in range(i)]  # rep: repeat
        mediaboxes = rep(ma, 10) + rep(mb, 10)
        cropboxes = rep(ca, 3) + rep(cb, 7) + rep(cc, 4) + rep(cd, 6)

        doc = None
        return slash._ImgGroup(doc, mediaboxes, cropboxes)

    g = build_imggroup()
    input_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    data = list(g.get(input_indices, kind='group'))
    assert len(data) == 2
    meta, indices, array = data[1]
    assert len(indices) == 5
    assert indices[0] == 11
    assert indices[1] == 13

    data = list(g.get(input_indices, kind='subgroup'))
    assert len(data) == 4
    meta, indices, array = data[1]
    assert len(indices) == 4
    assert indices[0] == 3
    assert indices[1] == 5
