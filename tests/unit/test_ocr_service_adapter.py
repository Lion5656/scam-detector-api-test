from backend.services.image_price_service.domain.models import OCRDocument
from backend.services.image_price_service.ocr.ocr_service import (
    extract_ocr_document,
)


class _TextOnlyOCRService:
    def extract_text(self, data: bytes) -> str:
        return "legacy OCR text"


class _DocumentOCRService:
    def extract_document(self, data: bytes) -> OCRDocument:
        return OCRDocument(text="document OCR text", width=100, height=200)


def test_extract_ocr_document_adapts_text_only_service():
    document = extract_ocr_document(_TextOnlyOCRService(), b"image")

    assert document == OCRDocument(text="legacy OCR text")


def test_extract_ocr_document_preserves_document_metadata():
    document = extract_ocr_document(_DocumentOCRService(), b"image")

    assert document.text == "document OCR text"
    assert document.width == 100
    assert document.height == 200
