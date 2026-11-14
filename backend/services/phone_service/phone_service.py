import logging

from backend.api.schemas.phone import PhoneReportError
from backend.repository.phone_repository import phone_repository
from backend.services.phone_service.phone_family import MIN_LINK_SCORE, analyze_family
from backend.utils.phone_normalizer import normalize

logger = logging.getLogger(__name__)


class PhoneService:
    @staticmethod
    def _display_phone(value: str | None) -> str | None:
        if value is None:
            return None
        return normalize(value) or None

    @staticmethod
    def _db_status_to_api_status(value: str | None) -> str:
        normalized = (value or "").strip().lower()
        if normalized in ("black", "white"):
            return normalized
        return "unknown"

    @staticmethod
    def _format_datetime(value) -> str | None:
        if value is None or isinstance(value, str):
            return value
        return value.strftime("%Y-%m-%d %H:%M:%S")

    def _list_family_links(self, target_phone: str, target_phone_type: str) -> list[dict]:
        """列出黑名單號碼的直接家族關聯；分析失敗時不影響查詢結果，回傳空清單。"""
        try:
            family = analyze_family(target_phone)
        except Exception:
            logger.warning("詐騙家族關聯分析失敗 phone=%s", target_phone, exc_info=True)
            return []

        links: list[dict] = []
        seen: set[tuple] = set()
        for item in family.get("static", []):
            weight = item.get("weight") or 0
            related_phone = self._display_phone(item.get("related_phone"))
            if not related_phone or weight < MIN_LINK_SCORE:
                continue
            reason = item.get("link_reason") or "同一家族成員"
            dedupe_key = (related_phone, weight, reason)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            links.append({
                "related_phone": related_phone,
                "weight": weight,
                "reason": reason,
                "target_phone_type": target_phone_type,
            })
        return links

    def query_phone(self, phone_number: str) -> dict:
        result = phone_repository.get_phone_record(phone_number)
        if not result:
            return {
                "phone_number": phone_number,
                "status": "unknown",
                "phone_type": None,
                "total_reports": None,
                "first_reported_at": None,
                "last_reported_at": None,
                "owner_name": None,
                "can_report": True,
                "family_static": [],
            }

        status = self._db_status_to_api_status(result.get("tags"))
        phone_type = result.get("phone_type") or result.get("dominant_tag")
        record_phone = result.get("phone_number") or phone_number

        family_static = []
        if status == "black":
            target_phone = normalize(record_phone) or normalize(phone_number)
            family_static = self._list_family_links(target_phone, phone_type or "unknown")

        return {
            "phone_number": self._display_phone(record_phone) or phone_number,
            "status": status,
            "phone_type": phone_type,
            "total_reports": int(result.get("total_reports") or 0) if result.get("total_reports") else None,
            "first_reported_at": self._format_datetime(result.get("first_reported_at")),
            "last_reported_at": self._format_datetime(result.get("last_reported_at")),
            "owner_name": result.get("owner_name") or result.get("org_name"),
            "can_report": status != "white",
            "family_static": family_static,
        }

    def report_suspicious(
        self,
        phone_number: str,
        phone_type: str,
        other_type: str | None = None,
        connection_reason: str | None = None,
        reporter_phone: str | None = None,
        transfer_type: int | None = 0,
        transfer_content: str | None = None,
    ) -> dict:
        if phone_type == "其他":
            phone_type = other_type or phone_type

        reason_text = (connection_reason or "").strip()
        try:
            # 號碼正規化由 repository 負責
            result = phone_repository.report_suspicious(
                phone_number=phone_number,
                phone_type=phone_type,
                reporter_phone=reporter_phone,
                transfer_type=transfer_type,
                transfer_content=transfer_content,
            )
        except ValueError as exc:
            raise PhoneReportError(str(exc)) from exc

        message = "可疑電話號碼已回報，已新增或更新黑名單記錄。"
        if reason_text:
            message = f"{message} 關聯原因：{reason_text}"

        return {
            "phone_number": self._display_phone(result.get("phone_number")) or phone_number,
            "status": "black",
            "total_reports": int(result.get("total_reports") or 1),
            "report_time": result.get("report_time"),
            "message": message,
            "connection_reason": reason_text or None,
        }


phone_service = PhoneService()
