"""Command line demo: index books, ask 3 answerable and 2 unanswerable questions, write the theme map."""

from __future__ import annotations

import argparse

from scripture_rag.config import Settings
from scripture_rag.corpus import load_folder
from scripture_rag.pipeline import RagPipeline
from scripture_rag.topics import TopicModeler, export_theme_map

QUESTIONS = [
    "Who is my shepherd?",
    "How does hatred cease?",
    "What is the highest excellence like?",
    "What is the capital of France?",
    "How many moons does Jupiter have?",
]


def main() -> None:
    """Run the demo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/demo")
    ap.add_argument("--map", default="site/theme_map.html")
    args = ap.parse_args()
    pipe = RagPipeline(Settings.from_env())
    books = load_folder(pipe, args.corpus)
    if not books:
        raise SystemExit(f"No books in {args.corpus}. Try: python demo.py --corpus tests/fixtures/books")
    print(f"Loaded {books} ({len(pipe.chunks)} passages)\n")
    for q in QUESTIONS:
        ans = pipe.ask(q)
        print(f"Q: {q}\nA: {ans.text}\n   (top score {ans.top_score:.2f}{', ' + ans.reason if ans.refused else ''})\n")
    ids, mat = pipe.embeddings()
    texts = [pipe.chunks[i].text for i in ids]
    res = TopicModeler().fit(mat, texts)
    print("Theme map:", export_theme_map(res, [pipe.chunks[i].book_id for i in ids], texts, args.map))


if __name__ == "__main__":
    main()
