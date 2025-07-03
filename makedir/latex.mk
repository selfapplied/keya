# latexmk pattern.
# "watch" targets regenerates the paper when the source changes.
# make dependencies generated using .fls output file from latexmk.

.out/%.pdf: tex_flags = -quiet -outdir=.out -pdf
.out/%.pdf: tex_i_flags ?= -interaction=nonstopmode
.out/%.pdf .out/%.fls : %.tex
	@: mkdir -p .out
	latexmk $(tex_flags) $(tex_i_flags) $< || cat .out/$*.log

.out/%.mk: .out/%.fls
	@echo Generating make dependencies $*.mk.
	@grep '^INPUT \./.*\.tex$$' .out/$*.fls | cut -d ' ' -f2 | \
	while read -r f; do \
		[ -f "$$f" ] && echo ".out/$*.pdf: $$f"; \
	done | sort -u -o .out/$*.mk

-include .out/make.mk
	
ifeq ($(suffix $(.DEFAULT_GOAL)), .pdf)
.PHONY: watch
watch: export tex_i_flags = -pvc -f
watch: ; ${MAKE} -B
endif

clean: ; latexmk -quiet -C .out/$(.DEFAULT_GOAL) -outdir=.out
