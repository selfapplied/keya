# Pattern to think about generated file targets as flow in one direction,
# and cleanup of the targets as reverse-flow.

# PHONY targets are actions you can run related to the file graph.
# all is default target
# Generated files go in .out/ directories

subdirs = $(dir $(wildcard */Makefile))
.PHONY: all clean $(subdirs)
all: $(subdirs)
clean: $(subdirs) $(wildcard */.out/*-)

# --- Subdirectory targets -------------------------------
# Implicit rules for subdirectories with Makefiles

submake=$(MAKE) $(MAKECMDGOALS) -C $*
$(subdirs): %: | %/Makefile; @$(submake)

# For a built target, you can generate a ".out/*-" file with a manifest for file cleanup.
# The minus file is a signifier for reverse flow, the removal of generated files.
# The manifest pattern is preferred as to rm -rf .out/.
# Example: .out/graphs.py- contents:
#   graph.png
#   othergraph.png

ifeq ($(.DEFAULT_GOAL), "clean")
.out/%-::
	@grep -Ev '^\s*($|#)' .out/$*.out 2> /dev/null | sed 's|^|.out/|' | xargs -r rm -v; exit 0
	@: rm -vf .out/$*.out 
	@rmdir .out

$(subdirs): %: | %/Makefile
	@${make} -C $@ clean
endif
