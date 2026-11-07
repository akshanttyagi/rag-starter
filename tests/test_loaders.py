from pathlib import Path

import docx
import pytest

from app.loaders import UnsupportedFileType, load, strip_repeated_lines

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_txt(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Hello.\n\nWorld.", encoding="utf-8")
    doc = load(f)
    assert doc.pages[0].text == "Hello.\n\nWorld."
    assert doc.pages[0].number is None


@pytest.mark.parametrize(
    "first_line,title",
    [
        ("# Release notes", "Release notes"),
        ("THE CONSTITUTION OF THE UNITED STATES", "The Constitution of the United States"),
        ("x" * 81, "my file"),
    ],
)
def test_txt_title(tmp_path, first_line, title):
    f = tmp_path / "my_file.md"
    f.write_text(f"{first_line}\n\nBody.", encoding="utf-8")
    assert load(f).title == title


def test_docx_paragraphs_and_tables(tmp_path):
    d = docx.Document()
    d.core_properties.title = "Policy"
    d.add_heading("Leave", level=1)
    d.add_paragraph("Staff get 20 days.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Sick leave"
    table.rows[0].cells[1].text = "10 days"
    f = tmp_path / "policy.docx"
    d.save(f)

    doc = load(f)
    assert doc.title == "Policy"
    assert doc.pages[0].text == "Leave\n\nStaff get 20 days.\n\nSick leave | 10 days"


def test_unsupported_extension(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFileType):
        load(f)


def test_running_headers_and_page_numbers_are_removed():
    topics = ["budget", "hiring", "security", "travel", "training", "audit"]
    pages = [
        f"Annual Report 2024\nThis page covers {t}.\nDetails on {t} follow.\nPage {i}"
        for i, t in enumerate(topics, start=1)
    ]
    cleaned = strip_repeated_lines(pages)
    assert cleaned[0] == "This page covers budget.\nDetails on budget follow."
    assert all("Annual Report" not in p and "Page" not in p for p in cleaned)


def test_short_documents_are_left_alone():
    pages = ["Header\nA", "Header\nB"]
    assert strip_repeated_lines(pages) == pages


def test_sample_pdf_has_pages_and_no_running_header():
    doc = load(SAMPLES / "nist-csf-2.0.pdf")
    assert doc.title == "The NIST Cybersecurity Framework (CSF) 2.0"
    assert len(doc.pages) == 32
    assert doc.pages[5].number == 6
    assert all("NIST CSWP 29" not in p.text.splitlines()[0] for p in doc.pages[2:])
