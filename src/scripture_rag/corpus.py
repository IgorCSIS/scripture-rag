"""Helpers to load a folder of books into a pipeline."""

from __future__ import annotations

from pathlib import Path

from .ingest import load_path
from .pipeline import RagPipeline


def load_folder(pipeline: RagPipeline, folder: str | Path) -> list[str]:
    """Load every PDF and TXT file in a folder.

    Parameters
    ----------
    pipeline : RagPipeline
        Target pipeline.
    folder : str or Path
        Directory to scan (not recursive).

    Returns
    -------
    list[str]
        Book ids loaded.
    """
    loaded = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".pdf", ".txt"}:
            doc = load_path(path)
            pipeline.add_document(doc)
            loaded.append(doc.book_id)
    return loaded
