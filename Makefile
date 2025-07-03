include makedir/biflow.mk

%.py::
	@:test -f $@ && exit 0
	uv init --script $@

deps:
	uv sync
