
Note
====

Design
------

This is the program to manually crop PDF margins,
from visual information of merged (superinterposed) images, in GUI.

So, while the program has many interpreter commands,
they are rather complementary. Especially,

``auto``:
    This is actually the first part of the code I've written,
    but I don't see any good place for it, in general workflow.

box edit commands (``append``, ``overwrite``, ``modify``, ``discard``, ``clear``):
    They are there, mainly for consistency with GUI commands.

    They are sometimes useful,
    but I don't plan to make them particularly easier to use.

    The main purpose I see is to be able to
    'replay' ``export`` command output.

---

Thanks to ``MuPDF`` and ``numpy``, 
This Python program is quite faster than ``briss`` (Java).

I'd like to know if there are any PDF-crop programs with this merging feature,
other than ``briss``.


Box Conversion
--------------

I think there are two complications
in MuPDF-PyMuPDF main APIs for box processing,
and while I tried to mend them a bit,
I'm not sure it adds much value, compared to extra trouble.

* MuPDF-PyMuPDF main APIs use float numbers,
  away from PDF real number strings.

* MuPDF-PyMuPDF reverses y-axis direction,
  making the bottom of MediaBox, to top.

For example, take the first page of ``PDFUA-Ref-2-05_BookChapter-german.pdf``
(in `PDF/UA Reference Suite 1.1 <https://www.pdfa.org/resource/pdfua-reference-suite/>`__).

It only defines ``MediaBox``:

    ``[0 0 595.276 841.89]``

It becomes in MuPDF-PyMuPDF:

    ``Rect(0.0, 0.0, 595.2760009765625, 841.8900146484375)``

When setting cropbox ``(10, 10, 585, 831)`` by ``fitz.set_cropbox``:

    ``Rect(10.0, 10.0, 585.0, 831.0)``

When save to file (note second number, ``10.89``):

    ``[10 10.89 585 831.89]``


``pdfslash`` adjusts this ``0.89``,
so running the crop command ``'crop 1 10,10,585,831'``:

    ``(10, 10.89, 585, 831.89)``

    ``[10 10 585 831]``
