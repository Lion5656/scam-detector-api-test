from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.persistence.mysql_connection import engine as db_engine
from backend.services.phone_service.phone_family import normalize, normalize_db_phone


class PhoneRepository:
    def __init__(self, engine: Any | None = None):
        self._engine = engine or db_engine

    def get_phone_record(self, phone_number: str) -> dict[str, Any] | None:
        if self._engine is None:
            return None

        normalized_phone = normalize(phone_number)
        db_phone = normalize_db_phone(phone_number)
        query = text(
            """
            SELECT p.phone_number,
                   p.tags,
                   b.report_count AS total_reports,
                   b.first_seen_at AS first_reported_at,
                   b.last_seen_at AS last_reported_at,
                   b.dominant_tag AS phone_type,
                   w.org_name AS owner_name
            FROM phone p
            LEFT JOIN fraud_blacklist b ON p.phone_number = b.phone_number
            LEFT JOIN white_list w ON p.phone_number = w.phone_number
            WHERE p.phone_number IN (:phone_db, :phone_full)
            """
        )

        try:
            with self._engine.begin() as conn:
                return conn.execute(
                    query,
                    {"phone_db": db_phone, "phone_full": normalized_phone},
                ).mappings().first()
        except OperationalError:
            return None

    def report_suspicious(
        self,
        phone_number: str,
        phone_type: str,
        reporter_phone: str | None,
        transfer_type: int | None = 0,
        transfer_content: str | None = None,
    ) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("資料庫連線失敗：database engine unavailable")

        normalized_phone = normalize(phone_number)
        db_phone = normalize_db_phone(phone_number)
        reporter_phone = normalize(reporter_phone) if reporter_phone else None
        transfer_content = (
            normalize(transfer_content)
            if transfer_type == 1 and transfer_content
            else transfer_content
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with self._engine.begin() as conn:
                phone_row = conn.execute(
                    text(
                        "SELECT phone_number, tags FROM phone WHERE phone_number IN (:phone_db, :phone_full)"
                    ),
                    {"phone_db": db_phone, "phone_full": normalized_phone},
                ).mappings().first()

                if phone_row and phone_row["tags"] == "White":
                    raise ValueError("該號碼已存在於白名單，無法回報為可疑號碼。")

                if phone_row:
                    conn.execute(
                        text(
                            "UPDATE phone SET tags = 'Black' WHERE phone_number IN (:phone_db, :phone_full)"
                        ),
                        {"phone_db": db_phone, "phone_full": normalized_phone},
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO phone(phone_number, tags) VALUES (:phone_number, 'Black')"
                        ),
                        {"phone_number": db_phone},
                    )

                blacklist_row = conn.execute(
                    text(
                        "SELECT report_count, dominant_tag FROM fraud_blacklist WHERE phone_number IN (:phone_db, :phone_full)"
                    ),
                    {"phone_db": db_phone, "phone_full": normalized_phone},
                ).mappings().first()

                if blacklist_row:
                    conn.execute(
                        text(
                            "UPDATE fraud_blacklist "
                            "SET report_count = report_count + 1, last_seen_at = :last_seen_at, "
                            "dominant_tag = :phone_type, referral_type = :referral_type, referral_info = :referral_info "
                            "WHERE phone_number = :phone_number"
                        ),
                        {
                            "last_seen_at": now,
                            "phone_type": phone_type,
                            "referral_type": transfer_type or 0,
                            "referral_info": transfer_content,
                            "phone_number": db_phone,
                        },
                    )
                    total_reports = int(blacklist_row["report_count"]) + 1
                else:
                    conn.execute(
                        text(
                            "INSERT INTO fraud_blacklist(phone_number, report_count, first_seen_at, last_seen_at, dominant_tag, referral_type, referral_info) "
                            "VALUES (:phone_number, 1, :first_seen_at, :last_seen_at, :phone_type, :referral_type, :referral_info)"
                        ),
                        {
                            "phone_number": db_phone,
                            "first_seen_at": now,
                            "last_seen_at": now,
                            "phone_type": phone_type,
                            "referral_type": transfer_type or 0,
                            "referral_info": transfer_content,
                        },
                    )
                    total_reports = 1

                conn.execute(
                    text(
                        "INSERT INTO fraud_reports(phone_number, reporter_phone, report_time, tag, referral_type, referral_info) "
                        "VALUES (:phone_number, :reporter_phone, :report_time, :tag, :referral_type, :referral_info)"
                    ),
                    {
                        "phone_number": db_phone,
                        "reporter_phone": reporter_phone and normalize_db_phone(reporter_phone),
                        "report_time": now,
                        "tag": phone_type,
                        "referral_type": transfer_type or 0,
                        "referral_info": transfer_content,
                    },
                )
        except OperationalError as exc:
            raise RuntimeError(f"資料庫連線失敗：{exc}") from exc

        return {
            "phone_number": normalized_phone,
            "status": "black",
            "total_reports": total_reports,
            "report_time": now,
        }


phone_repository = PhoneRepository()
