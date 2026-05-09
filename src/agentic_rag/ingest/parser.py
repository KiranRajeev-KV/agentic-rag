from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    status: str
    error: str | None
    conversion: object | None


class DoclingParser:
    def __init__(self) -> None:
        self._converter = DocumentConverter()
        self.parser_name = "docling"

    def parse_pdf(self, pdf_path: Path) -> ParseResult:
        try:
            conversion = self._converter.convert(str(pdf_path), raises_on_error=False)
        except Exception as err:  # noqa: BLE001
            return ParseResult(ok=False, status="failure", error=str(err), conversion=None)

        if conversion.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            return ParseResult(
                ok=True, status=conversion.status.value, error=None, conversion=conversion
            )
        return ParseResult(
            ok=False,
            status=conversion.status.value,
            error=f"docling status={conversion.status.value}",
            conversion=conversion,
        )
