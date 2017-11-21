# Simple makefile

.PHONY = all clean clean-cython test trailing-spaces

PYTHON ?= python
NOSETESTS ?= nosetests

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build/
	find . -name "*.pyc" | xargs rm -f
	find . -name "*.so" | xargs rm -f

clean-cython:
	find . -name "*.c" | egrep -v "randomkit.c|distributions.c" | xargs rm -f
	find . -name "*.cpp" | xargs rm -f

test:
	$(NOSETESTS)

trailing-spaces:
	find -name "*.py" | xargs sed 's/^M$$//'
