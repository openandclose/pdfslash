
"""Test scaling."""

import numpy

try:
    import PIL.Image as Image
except ImportError:
    Image = None

import pdfslash.slash as slash


def pillow_scale(img, shape, resample=None):
    resample = resample or Image.NEAREST
    w, h = shape[1], shape[0]
    return numpy.array(Image.fromarray(img).resize((w, h), resample))


def create_border():
    img = numpy.zeros(90).reshape((10, 9)).astype(numpy.uint8)
    img[[0, 4, 5, 9], :] = 255
    img[:, [0, 4, 8]] = 255
    return img


def create_cross():
    img = numpy.zeros(81).reshape((9, 9)).astype(numpy.uint8)
    for i in range(9):
        img[i, i] = 255
        img[i, 8 - i] = 255
    return img


def print_imgs(img, scale, a, b):
    print('--------------------')

    points = tuple(x * scale for x in tuple(img.shape))
    print('scale', scale, '(points', points, ')')
    print()

    print('first', a.shape)
    print(a)
    print()

    print('second', b.shape)
    print(b)
    print()


def test_scale_and_unscale():
    scaling = slash._Scaling()
    cnt1, cnt2 = 0, 0
    for s in [500, 501] + list(range(502, 2001, 101)) + [1998, 1999, 2000]:
        scale = round(s / 1000, 3)  # 0.5 to 2.0
        if scale == 0:
            continue
        scaling._set(scale)
        for i in range(101, 200):
            diff = i - scaling.get_unscaled(scaling.get_scaled((i,)))[0]
            assert diff in (0, 1, 2)
            diff = i - scaling.get_scaled(scaling.get_unscaled((i,)))[0]
            assert diff in (0, 1, 2)

            if 1 == diff:
                cnt1 += 1
            if 2 == diff:
                cnt2 += 1
    print('1: %d, 2: %d' % (cnt1, cnt2))


def test_scale_img():
    scale_img = slash.scale_img
    scaling = slash._Scaling()
    scales = 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 1.1, 1.3, 1.7, 2.0, 4.3
    imgs = create_border(), create_cross()

    for img in imgs:
        for scale in scales:
            s = scale_img(img, scale)
            scaling._set(scale)
            assert s.shape == scaling.get_scaled(img.shape)

            # it is better for the data to be the same as pillow's nn resize,
            # but it is not a requirement.
            if Image is None:
                continue
            p = pillow_scale(img, s.shape)
            if not numpy.array_equal(s, p):
                print_imgs(img, scale, p, s)
