"""FR-01 upload validation tests."""

import pytest

from scripture_rag.sanitize import UploadError, escape_for_display, safe_book_id, strip_control_chars, validate_upload


def test_valid_txt_and_pdf() -> None:
    assert validate_upload("a.txt", b"hello") == "txt"
    assert validate_upload("a.PDF", b"%PDF-1.7 rest") == "pdf"


@pytest.mark.parametrize(
    ("name", "data", "msg"),
    [
        ("a.exe", b"MZ", "Unsupported"),
        ("noext", b"x", "Unsupported"),
        ("fake.pdf", b"not a pdf", "not a PDF"),
        ("fake.txt", b"%PDF-1.4", "is a PDF"),
        ("bad.txt", b"\xff\xfe\xfa", "UTF-8"),
        ("empty.txt", b"", "empty"),
    ],
)
def test_rejections(name: str, data: bytes, msg: str) -> None:
    with pytest.raises(UploadError, match=msg):
        validate_upload(name, data)


def test_oversize() -> None:
    with pytest.raises(UploadError, match="larger"):
        validate_upload("big.txt", b"a" * (1024 * 1024 + 1), max_mb=1)


def test_helpers() -> None:
    assert strip_control_chars("a\x00b\x1bc\n\fd") == "abc\n\fd"
    assert escape_for_display("<script>") == "&lt;script&gt;"
    assert safe_book_id("My Book (1881).pdf") == "my_book_1881"
    assert safe_book_id("???.txt") == "book"
