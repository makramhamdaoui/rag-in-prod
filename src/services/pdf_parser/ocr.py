import io
import logging
from typing import Optional

import pytesseract
from PIL import Image
from PyPDF2 import PageObject, PdfReader

from src.services.chunking.chunker import clean_text
from src.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def extract_text_from_images(page: PageObject) -> str:
    """Use OCR to extract text from images embedded in a PDF page."""
    text = ""
    for image_file_object in page.images:
        try:
            image = Image.open(io.BytesIO(image_file_object.data))
            text += pytesseract.image_to_string(image)
            logger.info("Extracted text from image using OCR.")
        except Exception as e:
            logger.error(f"OCR error: {e}")
    return text


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract and clean text from a PDF file.
    Falls back to OCR for pages with no extractable text.
    """
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        logger.info(f"Opened PDF: {file_path} ({len(reader.pages)} pages)")
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logger.info(f"Page {page_num}: no text, attempting OCR.")
                    text += extract_text_from_images(page)
            except Exception as e:
                logger.error(f"Error on page {page_num}: {e}")
    return clean_text(text)
