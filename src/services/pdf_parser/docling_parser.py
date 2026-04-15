import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.services.pdf_parser.models import ParsedDocument, Section

logger = logging.getLogger(__name__)

# header labels docling uses
HEADER_LABELS = {"title", "section_header", "page_header"}


class DoclingPDFParser:
    """
    Structured PDF parser using Docling.
    Extracts sections with titles, content and page numbers.
    Falls back to PyPDF2 if Docling fails.
    """

    def __init__(
        self,
        do_ocr: bool = False,
        do_table_structure: bool = False,
        max_pages: int = 100,
        max_file_size_mb: int = 50,
    ):
        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        logger.info("DoclingPDFParser initialized.")

    def parse(self, pdf_path: str) -> Optional[ParsedDocument]:
        """
        Parse a PDF and return structured sections with metadata.

        Args:
            pdf_path: path to the PDF file

        Returns:
            ParsedDocument with sections, raw_text, num_pages
        """
        path = Path(pdf_path)

        # basic validation
        if not path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None
        if path.stat().st_size == 0:
            logger.error(f"PDF is empty: {pdf_path}")
            return None
        if path.stat().st_size > self.max_file_size_bytes:
            logger.warning(f"PDF too large: {path.stat().st_size / 1024 / 1024:.1f}MB")
            return None

        try:
            result = self._converter.convert(
                str(path),
                max_num_pages=self.max_pages,
                max_file_size=self.max_file_size_bytes,
            )
            doc = result.document
            sections = self._extract_sections(doc)
            raw_text = doc.export_to_text()

            # estimate page count from elements
            pages = set()
            for el in doc.texts:
                if hasattr(el, "prov") and el.prov:
                    for p in el.prov:
                        if hasattr(p, "page_no"):
                            pages.add(p.page_no)
            num_pages = max(pages) if pages else 0

            logger.info(
                f"Parsed '{path.name}': {len(sections)} sections, "
                f"{num_pages} pages, {len(raw_text)} chars"
            )
            return ParsedDocument(
                sections=sections,
                raw_text=raw_text,
                num_pages=num_pages,
                metadata={"source": "docling", "file": path.name},
            )

        except Exception as e:
            logger.error(f"Docling failed on {pdf_path}: {e}")
            return self._fallback_parse(path)

    def _extract_sections(self, doc) -> list:
        """Walk document elements and group text under section headers."""
        sections = []
        current = {"title": "Introduction", "content": "", "page": 0, "level": 1}

        for el in doc.texts:
            label = getattr(el, "label", "")
            text = getattr(el, "text", "").strip()
            if not text:
                continue

            # get page number
            page = 0
            if hasattr(el, "prov") and el.prov:
                for p in el.prov:
                    if hasattr(p, "page_no"):
                        page = p.page_no
                        break

            if label in HEADER_LABELS:
                # save previous section
                if current["content"].strip():
                    sections.append(Section(
                        title=current["title"],
                        content=current["content"].strip(),
                        page_number=current["page"],
                        level=current["level"],
                    ))
                # start new section
                level = 2 if label == "section_header" else 1
                current = {"title": text, "content": "", "page": page, "level": level}
            else:
                current["content"] += text + "\n"

        # save last section
        if current["content"].strip():
            sections.append(Section(
                title=current["title"],
                content=current["content"].strip(),
                page_number=current["page"],
                level=current["level"],
            ))

        return sections

    def _fallback_parse(self, path: Path) -> Optional[ParsedDocument]:
        """Fallback to PyPDF2 if Docling fails."""
        try:
            from PyPDF2 import PdfReader
            import re

            logger.info(f"Falling back to PyPDF2 for {path.name}")
            text = ""
            with open(path, "rb") as f:
                reader = PdfReader(f)
                num_pages = len(reader.pages)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t

            # clean text
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
            text = re.sub(r"\n+", "\n", text)
            text = re.sub(r"[ \t]+", " ", text).strip()

            return ParsedDocument(
                sections=[Section(title="Content", content=text, page_number=0)],
                raw_text=text,
                num_pages=num_pages,
                metadata={"source": "pypdf2_fallback", "file": path.name},
            )
        except Exception as e:
            logger.error(f"Fallback parsing also failed: {e}")
            return None
