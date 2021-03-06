
Spec
====

Box
---

In this program,
Box values are always expressed as (left, top, right, bottom),
in the coordinates in which
rotation is already applied, y-descendant,
``MediaBox``'s left-top moved to (0, 0), floats clipped to integers.

User created boxes must also be integers.

In addition, they must be unique in a page.
Duplicate boxes (the same boxes in a page) are not possible.


Config
------

If there is a environment variable ``'PDFSLASH_DIR'``
and it is a directory,
or ``'$XDG_CONFIG_HOME/pdfslash'`` or ``'~/.config/pdfslash'``
is a directory,
the program registers it as user-directory.

If there is a user-directory,
and the system can use Python standard library
`readline <https://docs.python.org/3/library/readline.html>`__,
the program uses a file ``'.history'`` (automatically created)
for the readline's history file
(And ``.python_history`` for ``Python`` command).

If there is a file ``'pdfslash.ini'`` in the directory,
the program reads it and update the configuration.

The defaults are:

{{ _fromsource_conf }}


Commandline
-----------

{{ _fromsource_commandline }}


Interpreter
-----------

* Token separator is space,
  so any command or argument must not have spaces.

* When the command string starts with ``'#'``,
  it is ignored.

* When the command string starts with Python regex ``'\[[a-z]+\] '``,
  the matched part is stripped.

  (e.g. ``'[gui] crop 1 10,10,400,500'`` -> ``'crop 1 10,10,400,500'``).

* Page number syntax is as follows.

{{ _fromsource_nstr }}

* Box syntax is as follows.

{{ _fromsource_box }}


Commands
^^^^^^^^

* commands are case sensitive
  (e.g. ``Set`` and ``Python`` start with capital letters).

* When commands take *optional* page numbers and they are omitted,
  *selected* pages are used.

* Admittedly ``select``, ``unselect``, ``fix`` and ``unfix`` tend to get very confusing.

  But normally you don't have to think about them,
  until when you need them.

* Interpreter and GUI are using the same undo and redo stack data.

  So in interpreter, you can go all back to the initial state,
  through any changes done in GUI.
  But in GUI, undo is bound to the GUI invocation,
  you can't go back past the changes done in the current GUI.

{{ _fromsource_cmds }}

**crop**

Alias for ``append``.

**quit**

Alias for ``exit``.

**(EOF)**

Alias for ``exit``. Send actual ``EOF``.

**'|' (pipe)**

If any command has a string ``'|'``,
the output of the command is passed to the shell.

Intended for a few basic things. E.g.:

.. code-block:: none

    show 1-100 | grep 155

    show 1-100 | cat > log.txt

(Currently, in the shell command string after ``'|'``,
only ``'>'``, ``'>>'`` and ``'|'`` are considered
as shell special tokens.
All other special characters are quoted,
so they may not work as expected).


GUI
---

Info
^^^^

title bar and label show some information.

**title bar example**:

    .. code-block:: none

        pdfslash: 1-13,21 (110%) [copy]

    ``1-13,21``: current page numbers (in current group and current view).

    ``(110%)``: current image zoom (when 100%, it is omitted).

    ``[copy]``: string ``copy``, shown only when copy is pending (after key ``c``).

**label example**:

    .. code-block:: none

        1/3 both 595x841, sel: 100,100,400,500 (300x400, 1.333)

    ``1/3``: current group number (``1``) and the number of groups (``3``).

    ``both``: current view (``both``, ``odds``, or ``evens``).

    ``595x842``: current source mediabox size (GUI canvas size). ``left`` and ``top`` are always zeros (``0,0,595,841``).

    ``sel``: active box (either string ``'sel'`` or ``'box'``).

    ``100,100,400,500``: active box coordinates.

    ``300x400``: active box size

    ``1.333``: ratio of height / width of active box.


Keyboard
^^^^^^^^

{{ _fromsource_gui }}
