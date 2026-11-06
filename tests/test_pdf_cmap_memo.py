"""pypdf ToUnicode CMap memo (backend/documents.py::_install_cmap_memo).

The memo must never change extracted text, must be installed once, and must actually skip the
repeated CMap parsing (the reason it exists)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from backend import documents

ROOT = Path(__file__).resolve().parent.parent
LEC = ROOT / "documents" / "Mathematics" / "Lec-1.pdf"


@pytest.fixture(autouse=True)
def _installed():
    documents._install_cmap_memo()
    yield


def _stock_pages(data: bytes) -> list[str]:
    """Extract with pypdf's original CMap parser (memo bypassed)."""
    from pypdf import PdfReader, _cmap

    wrapped = _cmap._parse_to_unicode
    _cmap._parse_to_unicode = wrapped._sah_orig
    try:
        r = PdfReader(io.BytesIO(data))
        return [p.extract_text() or "" for p in r.pages]
    finally:
        _cmap._parse_to_unicode = wrapped


@pytest.mark.skipif(not LEC.is_file(), reason="bundled lecture PDF missing")
def test_memo_output_identical_to_stock_pypdf():
    data = LEC.read_bytes()
    stock = _stock_pages(data)
    memo_first = documents.extract_pdf_pages(data)
    memo_again = documents.extract_pdf_pages(data)       # second pass served from the memo
    assert memo_first == stock
    assert memo_again == stock
    assert any(t.strip() for t in stock)


def test_install_is_idempotent_and_fail_open():
    from pypdf import _cmap

    first = _cmap._parse_to_unicode
    documents._install_cmap_memo()
    documents._install_cmap_memo()
    assert _cmap._parse_to_unicode is first
    assert getattr(first, "_sah_memo", False)


def test_memo_returns_copies_not_shared_objects():
    from pypdf import _cmap
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    # A font whose ToUnicode CMap is the bundled-free minimal stream; the memo must hand out copies
    # so a caller mutating the result cannot poison later lookups.
    stream = StreamObject()
    stream.set_data(b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
                    b"1 begincodespacerange <00> <FF> endcodespacerange\n"
                    b"1 beginbfchar <41> <0041> endbfchar\nendcmap\n")
    ft = DictionaryObject({NameObject("/ToUnicode"): stream, NameObject("/Subtype"): NameObject("/Type1")})
    a_map, a_int = _cmap._parse_to_unicode(ft)
    a_map[999] = "poison"
    a_int.append(12345)
    b_map, b_int = _cmap._parse_to_unicode(ft)
    assert 999 not in b_map and 12345 not in b_int
    assert b_map == _cmap._parse_to_unicode._sah_orig(ft)[0]
