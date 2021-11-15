
Spec
====

General
-------

**rotation**:

The program treats rotation-applied PDF cropboxes as given coordinates.
Operations are always done in this coordinates, ignoring rotations
(until writing, when the program resolves rotations).


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
for the readline's history file.

If there is a file ``'pdfslash.ini'`` in the directory,
the program reads it and update the configuration.

The defaults are:

.. literalinclude:: _help/conf.txt
    :language: ini


Commandline
-----------

.. include:: _help/commandline.txt


Interpreter
-----------

Token separator is space,
so any command or argument must not have spaces.

Page numbers and Box have some special syntax.

Page numbers:

.. literalinclude:: _help/nstr.txt
    :language: none


Box:

.. literalinclude:: _help/box.txt
    :language: none


Commands
^^^^^^^^

* commands are case sensitive (e.g. ``Set`` and ``Python`` start with capital letters).

* When commands take *optional* page numbers and they are omitted,
  selected pages are used.

* Admittedly ``select``, ``unselect``, ``fix`` and ``unfix`` tend to get very confusing.

  But note that you can safely ignore them if you don't use them.

* Interpreter and GUI are using the same undo and redo stack data.

  So in interpreter, you can go all back to the initial state,
  through any changes done in GUI.
  But in GUI, undo is bound to the GUI invocation,
  you can't go back past the changes done in the current GUI.

.. include:: _help/cmds.txt


**'|' (pipe)**

If any command has a string ``'|'``,
the output of the command is passed to the shell.

Intended for a few basic things. E.g.:

.. code-block:: none

    info 1-100 | grep 155

    info 1-100 | cat > log.txt


GUI
---

.. literalinclude:: _help/gui.txt
    :language: none
