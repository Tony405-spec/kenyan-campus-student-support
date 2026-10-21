"""Build the editable, explicitly paginated Markdown report as a PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("Advanced_Python_Dropout_Prediction_Report.md")
OUTPUT = Path(__file__).with_name("Advanced_Python_Dropout_Prediction_Report.pdf")


def _inline(text: str) -> str:
    """Escape text and support minimal Markdown emphasis and inline code."""
    safe = html.escape(text)
    safe = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', safe)
    safe = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", safe)
    safe = re.sub(r"(https?://[^\s]+)", r'<link href="\1" color="#215d8f">\1</link>', safe)
    return safe


def _footer(canvas, document) -> None:
    """Print consistent page number and report title in the footer."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c7d1dc"))
    canvas.line(0.65 * inch, 0.5 * inch, 7.85 * inch, 0.5 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#586675"))
    canvas.drawString(0.65 * inch, 0.32 * inch, "CPP 4201 | Kenyan Campus Support and Outcomes")
    canvas.drawRightString(7.85 * inch, 0.32 * inch, f"Page {document.page}")
    canvas.restoreState()


def _make_page(block: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 5))
        elif line.startswith("[[IMAGE:"):
            path = ROOT / line[8:-2]
            image = Image(str(path))
            if path.name == "00_home.png":
                image._restrictSize(3.0 * inch, 1.9 * inch)
            else:
                image._restrictSize(3.8 * inch, 2.55 * inch)
            image.hAlign = "CENTER"
            story.extend([Spacer(1, 3), image, Spacer(1, 4)])
        elif line.startswith("# "):
            is_cover = block.startswith("# Kenyan Campus Student Support")
            story.append(
                Paragraph(_inline(line[2:]), styles["title"] if is_cover else styles["h1"])
            )
            story.append(Spacer(1, 8))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), styles["h2"]))
        elif line.startswith("**") and line.endswith("**"):
            story.append(Paragraph(_inline(line), styles["center"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(_inline(line), styles["reference"]))
        else:
            story.append(Paragraph(_inline(line), styles["body"]))
    return story


def build() -> None:
    """Generate the 20-page PDF from manually paginated Markdown sections."""
    text = SOURCE.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("<!-- PAGE -->") if block.strip()]
    if len(blocks) != 20:
        raise ValueError(f"Expected 20 source pages, found {len(blocks)}")
    styles = getSampleStyleSheet()
    styles_map = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=29,
            textColor=colors.HexColor("#17324d"),
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17324d"),
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#215d8f"),
            spaceBefore=5,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.3,
            textColor=colors.HexColor("#202a33"),
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "center": ParagraphStyle(
            "ReportCenter",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=4,
        ),
    }
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.7 * inch,
        title=("Kenyan Campus Student Support and Outcome Modelling"),
        author="Tony Kenga; Baraka Blessings; Linet Maina",
    )
    story: list[object] = []
    for index, block in enumerate(blocks):
        story.extend(_make_page(block, styles_map))
        if index < len(blocks) - 1:
            story.append(PageBreak())
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


if __name__ == "__main__":
    build()
