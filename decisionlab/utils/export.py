"""Export results as CSV, a multi-sheet Excel workbook and a PDF summary."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.models import RankingResult, WeightingResult


@dataclass
class ReportData:
    """Everything needed to export a complete analysis."""

    title: str
    criteria: pd.DataFrame          # columns: Criterion, Type, Unit, Description
    matrix: pd.DataFrame            # alternatives x criteria (raw values)
    weighting: WeightingResult
    ranking: RankingResult
    explanation: pd.DataFrame
    baseline: pd.DataFrame | None = None
    created: datetime = field(default_factory=datetime.now)

    @property
    def method_label(self) -> str:
        return f"{self.weighting.method} + {self.ranking.method}"


def weights_table(weighting: WeightingResult) -> pd.DataFrame:
    """Weights as a table with a percentage column."""
    return pd.DataFrame({
        "Criterion": weighting.weights.index,
        "Weight": weighting.weights.values,
        "Weight (%)": (weighting.weights.values * 100).round(2),
    })


def ranking_csv(ranking: RankingResult) -> bytes:
    return ranking.ranking.to_csv(index=False).encode("utf-8-sig")


def weights_csv(weighting: WeightingResult) -> bytes:
    return weights_table(weighting).to_csv(index=False).encode("utf-8-sig")


def _scalar_details(details: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(k, v) for k, v in details.items()
            if isinstance(v, (int, float, str, bool)) and not isinstance(v, pd.DataFrame)]


def _frame_details(details: dict[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for key, value in details.items():
        if isinstance(value, pd.DataFrame):
            out.append((key, value))
        elif isinstance(value, pd.Series):
            out.append((key, value.to_frame()))
    return out


def excel_report(report: ReportData) -> bytes:
    """Build an .xlsx workbook with one sheet per result block."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    buffer = io.BytesIO()
    summary_rows = [
        ("Decision problem", report.title),
        ("Created", report.created.strftime("%Y-%m-%d %H:%M")),
        ("Weighting method", report.weighting.method),
        ("Ranking method", report.ranking.method),
        ("Best alternative", str(report.ranking.ranking.iloc[0]["Alternative"])),
        ("Best score", float(report.ranking.ranking.iloc[0]["Score"])),
        ("Alternatives", report.matrix.shape[0]),
        ("Criteria", report.matrix.shape[1]),
    ]
    summary_rows += [(f"{report.weighting.method}: {k}", v)
                     for k, v in _scalar_details(report.weighting.details)]
    for msg in report.weighting.warnings + report.ranking.warnings:
        summary_rows.append(("Warning", msg))

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows, columns=["Item", "Value"]).to_excel(
            writer, sheet_name="Summary", index=False)
        report.ranking.ranking.to_excel(writer, sheet_name="Ranking", index=False)
        weights_table(report.weighting).to_excel(writer, sheet_name="Weights", index=False)
        report.criteria.to_excel(writer, sheet_name="Criteria", index=False)
        report.matrix.to_excel(writer, sheet_name="Decision matrix", index_label="Alternative")
        report.explanation.to_excel(writer, sheet_name="Explanation", index=False)
        if report.baseline is not None:
            report.baseline.to_excel(writer, sheet_name="Baseline comparison", index=False)
        _write_blocks(writer, "Weighting details", _frame_details(report.weighting.details))
        _write_blocks(writer, "Ranking details", _frame_details(report.ranking.details))

        header_fill = PatternFill("solid", fgColor="DCE8F8")
        for sheet in writer.book.worksheets:
            for column_cells in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in column_cells)
                sheet.column_dimensions[get_column_letter(column_cells[0].column)].width = \
                    min(max(12, width + 2), 70)
            if sheet.title not in ("Weighting details", "Ranking details"):
                for cell in sheet[1]:
                    cell.font = Font(bold=True)
                    cell.fill = header_fill
                    cell.alignment = Alignment(vertical="center")
    return buffer.getvalue()


def _write_blocks(writer: pd.ExcelWriter, sheet: str,
                  blocks: list[tuple[str, pd.DataFrame]]) -> None:
    """Stack several labelled tables vertically on one sheet."""
    from openpyxl.styles import Font

    row = 0
    if not blocks:
        pd.DataFrame({"Info": ["No intermediate matrices for this method."]}).to_excel(
            writer, sheet_name=sheet, index=False)
        return
    for name, frame in blocks:
        frame.to_excel(writer, sheet_name=sheet, startrow=row + 1)
        writer.sheets[sheet].cell(row=row + 1, column=1,
                                  value=name.replace("_", " ").title()).font = Font(bold=True, size=12)
        row += len(frame) + 4


# --------------------------------------------------------------------------- PDF

_GREEK = {"λ": "lambda", "ξ": "xi", "σ": "sigma", "Σ": "Sum", "−": "-", "→": "->",
          "↑": "(max)", "↓": "(min)", "≤": "<=", "≥": ">=", "—": "-", "–": "-",
          "🏆": "", "…": "...", "’": "'", "“": '"', "”": '"'}


def _latin1(text: Any) -> str:
    """Make text safe for the PDF core fonts (Latin-1 only)."""
    s = str(text)
    for src, dst in _GREEK.items():
        s = s.replace(src, dst)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def pdf_available() -> bool:
    try:
        import fpdf  # noqa: F401
    except ImportError:
        return False
    return True


def pdf_report(report: ReportData) -> bytes:
    """Build a concise PDF summary (requires the ``fpdf2`` package)."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _latin1("DecisionLab - Decision Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, _latin1(report.title), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _latin1(f"{report.method_label}  |  {report.created:%Y-%m-%d %H:%M}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    best = report.ranking.ranking.iloc[0]
    pdf.set_fill_color(220, 232, 248)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, _latin1(f"Recommended alternative: {best['Alternative']} "
                                 f"(score {best['Score']:.4f})"), fill=True,
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_fill_color(255, 255, 255)
    pdf.ln(3)

    _pdf_table(pdf, "Final ranking", report.ranking.ranking)
    _pdf_table(pdf, "Criteria weights", weights_table(report.weighting))
    scalars = _scalar_details(report.weighting.details)
    if scalars:
        _pdf_table(pdf, f"{report.weighting.method} diagnostics",
                   pd.DataFrame(scalars, columns=["Measure", "Value"]))
    criteria = report.criteria[["Criterion", "Type", "Unit"]]
    _pdf_table(pdf, "Criteria", criteria)
    _pdf_table(pdf, "Decision matrix", report.matrix.reset_index(names="Alternative"))
    _pdf_table(pdf, "Why each alternative scored as it did",
               report.explanation[["Rank", "Alternative", "Main strengths", "Main weaknesses"]])
    if report.baseline is not None:
        _pdf_table(pdf, "Comparison with equal-weights baseline", report.baseline)

    warnings = report.weighting.warnings + report.ranking.warnings
    if warnings:
        _pdf_heading(pdf, "Warnings")
        pdf.set_font("Helvetica", "", 9)
        for msg in warnings:
            pdf.multi_cell(0, 5, _latin1(f"- {msg}"), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def _pdf_heading(pdf: Any, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _latin1(text), new_x="LMARGIN", new_y="NEXT")


def _pdf_table(pdf: Any, heading: str, frame: pd.DataFrame, max_cols: int = 8) -> None:
    from fpdf.fonts import FontFace

    _pdf_heading(pdf, heading)
    frame = frame.iloc[:, :max_cols]
    pdf.set_font("Helvetica", "", 8)
    heading_style = FontFace(emphasis="BOLD", fill_color=(220, 232, 248))

    def fmt(v: Any, column: str) -> str:
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, float):
            if "%" in column:
                return f"{v:.2f}%"
            return f"{v:.4f}" if abs(v) < 1e5 else f"{v:.4g}"
        return _latin1(v)

    with pdf.table(text_align="LEFT", line_height=5, first_row_as_headings=True,
                   headings_style=heading_style) as table:
        header = table.row()
        for col in frame.columns:
            header.cell(_latin1(col))
        for _, values in frame.iterrows():
            row = table.row()
            for column, v in zip(frame.columns, values):
                row.cell(fmt(v, str(column)))
    pdf.ln(2)
