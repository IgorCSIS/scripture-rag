PY ?= python

install:
	$(PY) -m pip install -r requirements-dev.txt -e .

test:
	$(PY) -m pytest --cov --cov-report=term-missing
	node --test site/tests/

lint:
	$(PY) -m ruff check .

demo:
	$(PY) demo.py --corpus $(or $(CORPUS),data/demo)

eval:
	$(PY) -m scripture_rag.eval --corpus $(or $(CORPUS),tests/fixtures/books) --gold $(or $(GOLD),tests/fixtures/gold_fixture.jsonl)

bench:
	$(PY) bench/bench_complexity.py

app:
	$(PY) app_gradio.py

streamlit:
	streamlit run app_streamlit.py

site:
	$(PY) -m http.server 8000 --directory site

.PHONY: install test lint demo eval bench app streamlit site
