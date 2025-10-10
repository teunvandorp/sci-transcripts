.SECONDARY:
.DELETE_ON_ERROR:

all : _book/index.html

lecture-%.Rmd :src/lecture-%.md clean_for_rmd.py
	python3 clean_for_rmd.py $< > $@

_book/index.html : index.Rmd  _bookdown.yml lecture-1.md lecture-2.Rmd lecture-2b.Rmd # lecture-3.Rmd lecture-3b.Rmd lecture-4.Rmd lecture-5.Rmd
	Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"
