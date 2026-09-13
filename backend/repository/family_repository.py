from __future__ import annotations

from typing import Any

from sqlalchemy import text

from backend.persistence.mysql_connection import engine as db_engine
from backend.utils.phone_normalizer import normalize as _shared_normalize
from backend.utils.phone_normalizer import normalize_db_phone as _shared_normalize_db_phone


class FamilyRepository:
    def __init__(self, engine: Any | None = None):
        self._engine = engine or db_engine

    def load_fraud_report_events(self, engine: Any | None = None) -> dict[str, list[tuple[str, Any]]]:
        """Load reporter-to-phone event history from the database."""
        active_engine = engine or self._engine
        events: dict[str, list[tuple[str, Any]]] = {}

        if active_engine is None:
            return events

        try:
            with active_engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT reporter_phone, phone_number, report_time
                        FROM fraud_reports
                        WHERE reporter_phone IS NOT NULL
                          AND phone_number IS NOT NULL
                          AND report_time IS NOT NULL
                        """
                    )
                ).mappings().all()

                for row in rows:
                    reporter = normalize_phone(row.get("reporter_phone"))
                    phone = normalize_phone(row.get("phone_number"))
                    reported_at = row.get("report_time")
                    if reporter and phone and reported_at:
                        events.setdefault(reporter, []).append((phone, reported_at))

                return {key: sorted(value, key=lambda item: item[1]) for key, value in events.items()}
        except Exception:
            return {}

    @staticmethod
    def _normalize_phone(phone: Any) -> str:
        return _shared_normalize(phone)

    @staticmethod
    def _normalize_db_phone(phone: Any) -> str:
        """轉成資料庫常見格式：手機 10 碼去前導 0 存 9 碼，家用市話 9 碼去前導 0 存 8 碼。"""
        return _shared_normalize_db_phone(phone)


    @staticmethod
    def _normalize_phone_alias(phone: Any) -> str:
        return FamilyRepository._normalize_phone(phone)

    @staticmethod
    def _normalize_db_phone_alias(phone: Any) -> str:
        return FamilyRepository._normalize_db_phone(phone)

    def load_genealogy(self, phone_number: str, engine: Any | None = None) -> dict[str, Any]:
        """Load related scam-family data from the configured MySQL tables."""
        active_engine = engine or self._engine
        if active_engine is None:
            return {"error": "database engine unavailable"}

        normalized_phone = normalize_phone(phone_number)
        db_phone = normalize_db_phone(phone_number)
        if not normalized_phone:
            return {"error": "invalid phone number"}

        try:
            with active_engine.connect() as conn:
                self_row = conn.execute(
                    text(
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
                    ),
                    {"phone_db": db_phone, "phone_full": normalized_phone},
                ).mappings().first()

                phone_type = self_row["dominant_tag"] if self_row else "其他"

                related_rows = conn.execute(
                    text(
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
                    ),
                    {
                        "phone_db": db_phone,
                        "phone_full": normalized_phone,
                        "phone_type": phone_type,
                    },
                ).mappings().all()

                reporter_phone_rows = conn.execute(
                    text(
                        """
                        SELECT DISTINCT reporter_phone
                        FROM fraud_reports
                        WHERE phone_number IN (:phone_db, :phone_full)
                        LIMIT 10
                        """
                    ),
                    {"phone_db": db_phone, "phone_full": normalized_phone},
                ).mappings().all()

                reporter_phones = [
                    row["reporter_phone"] for row in reporter_phone_rows if row.get("reporter_phone")
                ]

                reporter_rows = []
                if reporter_phones:
                    reporter_rows = conn.execute(
                        text(
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
                        ),
                        {"reporter_phone_list": tuple(reporter_phones)},
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
                for entry in related_rows:
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
                for entry in reporter_rows:
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

                if not records:
                    return {
                        "summary": {
                            "input_records": 1,
                            "evaluated_records": 0,
                            "total_edges": 0,
                        },
                        "scored_pairs": [],
                        "families": [],
                        "edges": [],
                        "records": [],
                    }

                from backend.services.phone_service.phone_family import build_phone_genealogy

                genealogy = build_phone_genealogy(records)
                genealogy["records"] = records
                return genealogy
        except Exception as exc:
            return {"error": str(exc), "scored_pairs": [], "families": [], "edges": [], "records": []}


normalize_phone = FamilyRepository._normalize_phone_alias
normalize_db_phone = FamilyRepository._normalize_db_phone_alias


family_repository = FamilyRepository()
