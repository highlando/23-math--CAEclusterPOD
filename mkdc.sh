MDFILE=slides.md

pandoc $MDFILE -o index.html \
    --citeproc \
    --mathjax  \
    -t revealjs --slide-level=2 -s \
    -V theme=solarized \
    -V NavigationMode=linear \
    -V viewDistance=15 -V width=1280 -V height=880 -V margin=0.05

# vim -s jointhelines.vim slides.html
