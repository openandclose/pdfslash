
PHONIES = help test guitest html

.PHONY: $(PHONIES)

first: all

all: test guitest html
	@echo 'Success!'

help:
	@echo $(PHONIES)

test:
	pytest -x

guitest:
	pytest tests/_test_tkinter.py

html:
	docs/_helpgen.py
	$(MAKE) -C docs html
