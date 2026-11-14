from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import bindparam, text

from backend.persistence.mysql_connection import engine as db_engine
from backend.utils.phone_normalizer import normalize, normalize_db_phone

_SELF_ROW_QUERY = text(
    """
    SELECT phone_number,
           dominant_tag,
           first_seen_at,
           last_seen_at,
           referral_type,
           referral_info
    FROM blacklist
    WHERE phone_number IN (:phone_db, :phone_full)
    LIMIT 1
    """
)

_RELATED_ROWS_QUERY = text(
    """
    SELECT DISTINCT phone_number,
           dominant_tag AS phone_type,
           first_seen_at AS first_reported_at,
           last_seen_at AS last_reported_at,
           referral_type AS transfer_type,
           referral_info AS transfer_content
    FROM blacklist
    WHERE phone_number NOT IN (:phone_db, :phone_full)
      AND (
            referral_info IN (:phone_db, :phone_full)
            OR LEFT(phone_number, 6) = LEFT(:phone_db, 6)
            OR dominant_tag = :phone_type
          )
    ORDER BY
      CASE
        WHEN referral_info IN (:phone_db, :phone_full) THEN 0
        WHEN LEFT(phone_number, 6) = LEFT(:phone_db, 6) THEN 1
        ELSE 2
      END
    LIMIT 20
    """
)

_REPORTER_PHONES_QUERY = text(
    """
    SELECT DISTINCT reporter_phone
    FROM fraud_reports
    WHERE phone_number IN (:phone_db, :phone_full)
    LIMIT 10
    """
)

_REPORTER_ROWS_QUERY = text(
    """
    SELECT DISTINCT fr.phone_number,
           fr.tag AS phone_type,
           fr.report_time AS first_reported_at,
           fr.report_time AS last_reported_at,
           fr.referral_type AS transfer_type,
           fr.referral_info AS transfer_content
    FROM fraud_reports fr
    WHERE fr.reporter_phone IN :reporter_phone_list
    LIMIT 20
    """
).bindparams(bindparam("reporter_phone_list", expanding=True))

# 只撈指定號碼的通報事件；原本每次查詢都整張表載入，資料量越大越慢
_REPORT_EVENTS_QUERY = text(
    """
    SELECT reporter_phone, phone_number, report_time
    FROM fraud_reports
    WHERE phone_number IN :phone_list
      AND reporter_phone IS NOT NULL
      AND report_time IS NOT NULL
    """
).bindparams(bindparam("phone_list", expanding=True))


class FamilyRepository:
    def __init__(self, engine: Any | None = None):
        self._engine = engine or db_engine

    def list_fraud_report_events(
        self, phones: Iterable[str], engine: Any | None = None
    ) -> dict[str, list[tuple[str, Any]]]:
        """回傳 {通報者: [(被通報號碼, 通報時間), ...]}，只包含 phones 內號碼的事件。

        號碼同時以完整格式與資料庫格式（去前導 0）查詢，與本模組其他查詢的假設一致。
        """
        active_engine = engine or self._engine
        phone_list = sorted(
            {form for phone in phones for form in (phone, normalize_db_phone(phone)) if form}
        )
        if active_engine is None or not phone_list:
            return {}

        with active_engine.connect() as conn:
            rows = conn.execute(_REPORT_EVENTS_QUERY, {"phone_list": phone_list}).mappings().all()

        events: dict[str, list[tuple[str, Any]]] = {}
        for row in rows:
            reporter = normalize(row.get("reporter_phone"))
            phone = normalize(row.get("phone_number"))
            reported_at = row.get("report_time")
            if reporter and phone and reported_at:
                events.setdefault(reporter, []).append((phone, reported_at))
        return events

    def list_family_records(
        self, phone_number: str, engine: Any | None = None
    ) -> list[dict[str, Any]]:
        """撈出查詢號碼本身、可能相關的黑名單號碼與同通報者通報過的號碼，作為家族分析輸入。"""
        active_engine = engine or self._engine
        normalized_phone = normalize(phone_number)
        if active_engine is None or not normalized_phone:
            return []

        phone_params = {
            "phone_db": normalize_db_phone(phone_number),
            "phone_full": normalized_phone,
        }
        with active_engine.connect() as conn:
            self_row = conn.execute(_SELF_ROW_QUERY, phone_params).mappings().first()
            phone_type = self_row["dominant_tag"] if self_row else "其他"
            related_rows = conn.execute(
                _RELATED_ROWS_QUERY, {**phone_params, "phone_type": phone_type}
            ).mappings().all()

            reporter_phones = [
                row["reporter_phone"]
                for row in conn.execute(_REPORTER_PHONES_QUERY, phone_params).mappings().all()
                if row.get("reporter_phone")
            ]
            reporter_rows = []
            if reporter_phones:
                reporter_rows = conn.execute(
                    _REPORTER_ROWS_QUERY, {"reporter_phone_list": reporter_phones}
                ).mappings().all()

        records: list[dict[str, Any]] = []
        if self_row:
            records.append({
                "電話號碼": self_row["phone_number"],
                "通報標籤": self_row["dominant_tag"] or "其他",
                "首次通報時間": self_row["first_seen_at"],
                "最後通報時間": self_row["last_seen_at"],
                "轉介類型": self_row["referral_type"] or 0,
                "轉介內容": self_row["referral_info"] or "",
                "通報者識別碼": "",
                "通報次數": 1,
            })
        for entry in [*related_rows, *reporter_rows]:
            records.append({
                "電話號碼": entry["phone_number"],
                "通報標籤": entry["phone_type"] or "其他",
                "首次通報時間": entry["first_reported_at"],
                "最後通報時間": entry["last_reported_at"],
                "轉介類型": entry["transfer_type"] or 0,
                "轉介內容": entry["transfer_content"] or "",
                "通報者識別碼": "",
                "通報次數": 1,
            })
        return records


family_repository = FamilyRepository()
