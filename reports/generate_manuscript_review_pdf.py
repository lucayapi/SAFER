"""Render the prioritized manuscript review Markdown as a self-contained PDF."""

from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "manuscript_review_2026-09-29.md"
DESTINATION = ROOT / "manuscript_review_prioritized_2026-09-29.pdf"

PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4 in PostScript points
LEFT, TOP, BOTTOM = 48, 58, 48
BODY_SIZE, BODY_LEADING = 10.2, 13.4


def pdf_escape(value: str) -> bytes:
    encoded = value.encode("cp1252", errors="replace")
    return encoded.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def wrapped_lines(markdown: str) -> list[tuple[str, str]]:
    """Convert lightweight Markdown into styled, wrapped text lines."""
    output: list[tuple[str, str]] = []
    for raw in markdown.splitlines():
        if not raw.strip():
            output.append(("space", ""))
        elif raw.startswith("# "):
            output.append(("title", raw[2:].strip()))
        elif raw.startswith("## "):
            output.append(("heading", raw[3:].strip()))
        elif raw.startswith("### "):
            output.append(("subheading", raw[4:].strip()))
        elif raw.startswith("- "):
            for index, part in enumerate(
                textwrap.wrap(raw[2:].strip(), width=88, break_long_words=False)
            ):
                output.append(("bullet" if index == 0 else "bullet_cont", part))
        else:
            for part in textwrap.wrap(raw.strip(), width=96, break_long_words=False):
                output.append(("body", part))
    return output


def style_attributes(style: str) -> tuple[str, float, float, tuple[float, float, float], float]:
    if style == "title":
        return "F2", 18, 23, (0.10, 0.20, 0.34), 0
    if style == "heading":
        return "F2", 13, 18, (0.12, 0.30, 0.48), 0
    if style == "subheading":
        return "F2", 11, 15, (0.52, 0.12, 0.12), 0
    if style == "space":
        return "F1", BODY_SIZE, 7, (0, 0, 0), 0
    if style in {"bullet", "bullet_cont"}:
        return "F1", BODY_SIZE, BODY_LEADING, (0, 0, 0), 13
    return "F1", BODY_SIZE, BODY_LEADING, (0, 0, 0), 0


def build_pages(lines: list[tuple[str, str]]) -> list[list[tuple[str, str, float]]]:
    pages: list[list[tuple[str, str, float]]] = [[]]
    y = PAGE_HEIGHT - TOP
    for style, text in lines:
        _, _, leading, _, _ = style_attributes(style)
        before = 6 if style in {"heading", "subheading"} else 0
        if y - before - leading < BOTTOM:
            pages.append([])
            y = PAGE_HEIGHT - TOP
        y -= before
        pages[-1].append((style, text, y))
        y -= leading
    return pages


def text_command(
    text: str,
    x: float,
    y: float,
    font: str,
    size: float,
    color: tuple[float, float, float],
) -> bytes:
    red, green, blue = color
    return (
        f"BT /{font} {size} Tf {red:.3f} {green:.3f} {blue:.3f} rg "
        f"1 0 0 1 {x:.1f} {y:.1f} Tm ".encode()
        + b"(" + pdf_escape(text) + b") Tj ET\n"
    )


def page_stream(page: list[tuple[str, str, float]], page_number: int, page_count: int) -> bytes:
    fragments: list[bytes] = [
        text_command(
            "Prioritized manuscript review - carpentry and joinery application",
            LEFT, PAGE_HEIGHT - 30, "F1", 8, (0.35, 0.35, 0.35),
        ),
        text_command(
            f"29 September 2026   |   Page {page_number} of {page_count}",
            LEFT, 27, "F1", 8, (0.35, 0.35, 0.35),
        ),
    ]
    for style, text, y in page:
        font, size, _, color, indent = style_attributes(style)
        prefix = "- " if style == "bullet" else "  " if style == "bullet_cont" else ""
        fragments.append(text_command(prefix + text, LEFT + indent, y, font, size, color))
    return b"".join(fragments)


def make_pdf(pages: list[list[tuple[str, str, float]]]) -> bytes:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    page_numbers: list[int] = []
    for index, page in enumerate(pages):
        page_number = len(objects) + 1
        content_number = page_number + 1
        page_numbers.append(page_number)
        stream = page_stream(page, index + 1, len(pages))
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {content_number} 0 R >>"
            ).encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream")
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode()

    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, object_body in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode())
        result.extend(object_body)
        result.extend(b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    result.extend(b"0000000000 65535 f \n")
    result.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    result.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(result)


def main() -> None:
    pages = build_pages(wrapped_lines(SOURCE.read_text(encoding="utf-8")))
    DESTINATION.write_bytes(make_pdf(pages))
    print(f"Wrote {DESTINATION} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
