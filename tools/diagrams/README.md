# Diagrams

Source for every SVG diagram and the logo, so they can be rebuilt from scratch.

```bash
python tools/diagrams/gen.py    # docs/diagrams/*.svg and site/assets/{architecture,pipeline,retrieval,clustering}.svg
python tools/diagrams/logo.py   # site/assets/logo.svg and favicon.svg
```

Both scripts use only the Python standard library and write into the repo. `gen.py` prints `OVER` lines if any label overflows its box.

`render.sh in.svg out.png bg width [height]` renders an SVG to PNG with headless Chrome, for visual review.
