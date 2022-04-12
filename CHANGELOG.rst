
Changelog
=========

Unreleased
----------

---

v0.3.0 (2022-04-13)
-------------------

**Change:**

* Change and Fix box coords inconsistency [676dca64]

  PyMuPDF box coords are mediabox 'top' aligned,
  and the program normally used them,
  while img box coords are mediabox 'top-left' aligned.

  I have been using them very inconsistently.

  Now, all boxes in interfaces are img box type ('top-left' aligned).

**Add:**

* Add x key for exact coords paste (GUI) [80dde8d1]


v0.2.3 (2022-04-07)
-------------------

**Change:**

* Change overwritten 'h' key to 'H' (GUI help method) [5394c9c5]

  'h' is used in 'move rectangle' ('h', 'j', 'k', 'l').

  Before:
    h:  print help in terminal

  After:
    H:  print help in terminal

**Fix:**

* Fix `--nocheck` commandline argument (it didn't work previously) [ce1481e1]

* Fix pdf y-direction adjusting error [9e9144e6]


v0.2.2 (2022-02-05)
-------------------

* Fix a few bugs (info related)


v0.2.1 (2022-01-19)
-------------------

* Add note.rst


v0.2.0 (2022-01-17)
-------------------

* Done initial iteration
  (initial feature additions and design check)


v0.1.0 (2021-11-15)
-------------------

* Initial commit
