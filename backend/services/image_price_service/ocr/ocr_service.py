"""將商品圖片送往 Google Cloud Vision，並整理 OCR 文字與版面資訊。"""

import json
import os
import re
import unicodedata
from typing import Any, cast

from google.oauth2 import service_account

from backend.config import settings
from backend.services.image_price_service.models import OCRDocument, OCRTextBlock


class OCRService:
    """透過 Google Vision 擷取文字與版面資訊。"""
    @staticmethod
    def _clean_ocr_text(text: str, *, preserve_lines: bool = False) -> str:
        """整理 OCR 空白與常見英數辨識錯字。"""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\r\t]+", " ", text)
        if preserve_lines:
            text = "\n".join(
                re.sub(r"\s+", " ", line).strip()
                for line in text.splitlines()
                if line.strip()
            )
        else:
            text = re.sub(r"\s+", " ", text).strip()

        # 修正商品名稱與型號中常見的英數字元誤判。
        replacements = {
            "iph0ne": "iphone",
            "ga1axy": "galaxy",
            "airp0ds": "airpods",
            "rnacbook": "macbook",
        }
        cleaned = text
        for src, dst in replacements.items():
            cleaned = re.sub(src, dst, cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    def _extract_paragraph_text(paragraph) -> str:
        """依符號與單字順序組合 Google Vision 段落文字。"""
        words: list[str] = []
        for word in paragraph.words:
            word_text = "".join(symbol.text for symbol in word.symbols)
            if word_text:
                words.append(word_text)
        return " ".join(words).strip()

    @staticmethod
    def _bbox_from_vertices(vertices) -> tuple[float, float, float, float]:
        """將邊界框頂點換算為最小與最大座標。"""
        xs = [float(v.x or 0) for v in vertices]
        ys = [float(v.y or 0) for v in vertices]
        return min(xs), min(ys), max(xs), max(ys)

    def extract_text(self, data: bytes) -> str:
        """從圖片取得 OCR 全文。"""
        return self.extract_document(data).text

    def extract_document(self, data: bytes) -> OCRDocument:
        """從圖片建立含版面資訊的 OCR 文件。"""
        provider = settings.OCR_PROVIDER.strip().lower()
        if provider not in {"google", "google_vision", "gcv"}:
            raise RuntimeError("OCR_PROVIDER 僅支援 google_vision")

        return self._extract_document_with_google_vision(data)

    def _extract_text_with_google_vision(self, data: bytes) -> str:
        """透過 Google Vision 取得 OCR 全文。"""
        return self._extract_document_with_google_vision(data).text

    def _extract_document_with_google_vision(self, data: bytes) -> OCRDocument:
        """呼叫 Google Vision 並整理 OCR 文件。"""
        try:
            from google.cloud import vision
        except ImportError as e:
            raise RuntimeError("Google Cloud Vision 套件未安裝，請安裝 google-cloud-vision") from e

        try:
            key_json_str = settings.GCP_OCR_SERVICE_ACCOUNT_JSON or os.getenv("GCP_OCR_SERVICE_ACCOUNT_JSON")
            key_info = json.loads(key_json_str) if key_json_str else None

            credentials = service_account.Credentials.from_service_account_info(key_info)
            client = vision.ImageAnnotatorClient(credentials=credentials)
            image = vision.Image(content=data)
            language_hints = [h.strip() for h in settings.GCV_LANGUAGE_HINTS.split(",") if h.strip()]
            image_context = vision.ImageContext(language_hints=language_hints)

            # 文件文字偵測能保留段落結構，較適合包含多行刊登資訊的截圖。
            doc_response = cast(Any, client).document_text_detection(image=image, image_context=image_context)
            if doc_response.error.message:
                raise RuntimeError(f"Google Vision OCR 錯誤: {doc_response.error.message}")
            doc_text = (doc_response.full_text_annotation.text or "").strip()
            if doc_text:
                pages = doc_response.full_text_annotation.pages
                blocks: list[OCRTextBlock] = []
                page_width = 0.0
                page_height = 0.0

                for page in pages:
                    page_width = max(page_width, float(page.width or 0))
                    page_height = max(page_height, float(page.height or 0))
                    for block in page.blocks:
                        for paragraph in block.paragraphs:
                            paragraph_text = self._extract_paragraph_text(paragraph)
                            if not paragraph_text:
                                continue
                            x0, y0, x1, y1 = self._bbox_from_vertices(
                                paragraph.bounding_box.vertices
                            )
                            confidence = getattr(paragraph, "confidence", None)
                            blocks.append(OCRTextBlock(
                                text=self._clean_ocr_text(paragraph_text),
                                x=x0,
                                y=y0,
                                width=x1 - x0,
                                height=y1 - y0,
                                confidence=float(confidence) if confidence is not None else None,
                            ))

                blocks.sort(key=lambda item: (
                    item.y if item.y is not None else float("inf"),
                    item.x if item.x is not None else float("inf"),
                ))
                return OCRDocument(
                    text=self._clean_ocr_text(doc_text, preserve_lines=True),
                    blocks=blocks,
                    width=page_width or None,
                    height=page_height or None,
                )

            response = cast(Any, client).text_detection(image=image, image_context=image_context)
        except Exception as exc:
            raise RuntimeError(f"Google Vision OCR 執行失敗: {exc}") from exc

        if response.error.message:
            raise RuntimeError(f"Google Vision OCR 錯誤: {response.error.message}")

        if not response.text_annotations:
            return OCRDocument(text="")

        return OCRDocument(
            text=self._clean_ocr_text(
                response.text_annotations[0].description,
                preserve_lines=True,
            )
        )


def extract_ocr_document(service: Any, data: bytes) -> OCRDocument:
    """取得 OCR 文件並相容僅支援全文的服務。"""
    extract_document = getattr(service, "extract_document", None)
    if callable(extract_document):
        return cast(OCRDocument, extract_document(data))
    return OCRDocument(text=service.extract_text(data))


ocr_service = OCRService()
