from backend.api.schemas.phone import PhoneReportError
from backend.repository.phone_repository import phone_repository
from backend.services.phone_service.phone_family import (
    normalize,
    analyze_family,
)


class PhoneService:
    def _build_relevant_families(self, phone_number: str, families: list, phone_type: str | None) -> list:
        """Keep only families that include the queried number and add direct connection context."""
        relevant_families: list[dict] = []
        for family in families:
            members = family.get("members", []) or []
            if phone_number not in members:
                continue

            family_context = dict(family)
            family_context["target_phone_type"] = phone_type or "unknown"
            family_context["connection_reason"] = (
                "同一家族成員且通報標籤一致"
                if family.get("primary_labels")
                else "同一家族成員"
            )
            relevant_families.append(family_context)
        return relevant_families

    def _build_family_summary(self, phone_number: str, genealogy: dict, phone_type: str | None = None) -> list[dict]:
        """Return only direct, accepted relationships whose score meets the threshold."""
        related: list[dict] = []
        seen: set[str] = set()

        for pair in genealogy.get("scored_pairs", []) or []:
            if not pair.get("passed"):
                continue

            score = int(pair.get("score") or 0)
            if score < 40:
                continue

            left = str(pair.get("left_phone") or "")
            right = str(pair.get("right_phone") or "")
            status = pair.get("status", "accepted")
            reason = "; ".join(pair.get("reasons", [])[:2]) if pair.get("reasons") else "同一家族成員"

            if left == phone_number and right and right != phone_number:
                if right in seen:
                    continue
                related.append({
                    "related_phone": right,
                    "weight": score,
                    "reason": reason,
                    "connection_reason": reason,
                    "target_phone_type": phone_type or "unknown",
                    "status": status,
                })
                seen.add(right)
            elif right == phone_number and left and left != phone_number:
                if left in seen:
                    continue
                related.append({
                    "related_phone": left,
                    "weight": score,
                    "reason": reason,
                    "connection_reason": reason,
                    "target_phone_type": phone_type or "unknown",
                    "status": status,
                })
                seen.add(left)

        return related[:10]
    
    @staticmethod
    @staticmethod
    def _display_phone(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize(value)
        return normalized or None

    @staticmethod
    def _db_status_to_api_status(value: str | None) -> str:
        normalized = (value or "").strip().lower()
        if normalized == "black":
            return "black"
        if normalized == "white":
            return "white"
        return "unknown"

    def query_phone(self, phone_number: str) -> dict:
        result = phone_repository.get_phone_record(phone_number)

        can_report = True
        if not result:
            return {
                "phone_number": phone_number,
                "status": "unknown",
                "phone_type": None,
                "total_reports": None,
                "first_reported_at": None,
                "last_reported_at": None,
                "owner_name": None,
                "can_report": can_report,
                "family_static": []
            }

        first_reported_at = (
            result.get("first_reported_at") 
            if isinstance(result.get("first_reported_at"), str)
            else (result.get("first_reported_at").strftime("%Y-%m-%d %H:%M:%S") if result.get("first_reported_at") else None)
        )
        last_reported_at = (
            result.get("last_reported_at")
            if isinstance(result.get("last_reported_at"), str)
            else (result.get("last_reported_at").strftime("%Y-%m-%d %H:%M:%S") if result.get("last_reported_at") else None)
        )

        status = self._db_status_to_api_status(result.get("tags"))
        if status == "white":
            can_report = False
        phone_type = result.get("phone_type")
        if not phone_type:
            phone_type = result.get("dominant_tag")
        
        # Build direct family evidence for blacklisted numbers
        family_static = []
        
        if status == "black":
            target_phone = normalize(result.get("phone_number") or phone_number) or normalize(phone_number)
            try:
                family = analyze_family(target_phone) or {}
                seen_family_links = set()
                family_static = []
                for item in family.get("static", []) or []:
                    related_phone = item.get("related_phone")
                    weight = item.get("weight") or 0
                    reason = item.get("link_reason") or "同一家族成員"
                    if not related_phone or weight < 40:
                        continue
                    display_related_phone = self._display_phone(related_phone)
                    if not display_related_phone:
                        continue
                    dedupe_key = (display_related_phone, weight, reason)
                    if dedupe_key in seen_family_links:
                        continue
                    seen_family_links.add(dedupe_key)
                    family_static.append({
                        "related_phone": display_related_phone,
                        "weight": weight,
                        "reason": reason,
                        "target_phone_type": result.get("phone_type") or phone_type or "unknown",
                    })
            except Exception:
                family = analyze_family(target_phone) or {}
                seen_family_links = set()
                family_static = []
                for item in family.get("static", []) or []:
                    related_phone = item.get("related_phone")
                    weight = item.get("weight") or 0
                    reason = item.get("link_reason") or "同一家族成員"
                    if not related_phone or weight < 40:
                        continue
                    display_related_phone = self._display_phone(related_phone)
                    if not display_related_phone:
                        continue
                    dedupe_key = (display_related_phone, weight, reason)
                    if dedupe_key in seen_family_links:
                        continue
                    seen_family_links.add(dedupe_key)
                    family_static.append({
                        "related_phone": display_related_phone,
                        "weight": weight,
                        "reason": reason,
                        "target_phone_type": result.get("phone_type") or phone_type or "unknown",
                    })

        display_phone_number = self._display_phone(result.get("phone_number") or phone_number)

        return {
            "phone_number": display_phone_number or phone_number,
            "status": status,
            "phone_type": phone_type,
            "total_reports": int(result.get("total_reports") or 0) if result.get("total_reports") else None,
            "first_reported_at": first_reported_at,
            "last_reported_at": last_reported_at,
            "owner_name": result.get("owner_name") or result.get("org_name"),
            "can_report": can_report,
            "family_static": family_static
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

        reporter_phone = normalize(reporter_phone) if reporter_phone else None
        transfer_content = normalize(transfer_content) if transfer_type == 1 and transfer_content else transfer_content
        reason_text = (connection_reason or "").strip()
        try:
            result = phone_repository.report_suspicious(
                phone_number=phone_number,
                phone_type=phone_type,
                reporter_phone=reporter_phone,
                transfer_type=transfer_type,
                transfer_content=transfer_content,
            )
        except ValueError as exc:
            raise PhoneReportError(str(exc)) from exc
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

        total_reports = int(result.get("total_reports") or 1)
        message = "可疑電話號碼已回報，已新增或更新黑名單記錄。"
        if reason_text:
            message = f"{message} 關聯原因：{reason_text}"

        return {
            "phone_number": self._display_phone(result.get("phone_number")) or phone_number,
            "status": "black",
            "total_reports": total_reports,
            "report_time": result.get("report_time"),
            "message": message,
            "connection_reason": reason_text or None,
        }


phone_service = PhoneService()
