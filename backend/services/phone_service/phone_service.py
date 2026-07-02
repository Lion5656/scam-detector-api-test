from datetime import datetime

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.database import engine


class PhoneService:
    def query_phone(self, phone_number: str) -> dict:
        query = text(
            """
            SELECT p.phone_number,
                   p.status,
                   p.phone_type,
                   b.total_reports,
                   b.first_reported_at,
                   b.last_reported_at,
                   w.owner_name
            FROM phone p
            LEFT JOIN blacklist b ON p.id = b.phone_id
            LEFT JOIN whitelist w ON p.id = w.phone_id
            WHERE p.phone_number = :phone_number
            """
        )

        try:
            with engine.begin() as conn:
                result = conn.execute(query, {"phone_number": phone_number}).mappings().first()
        except OperationalError as exc:
            raise RuntimeError(f"資料庫連線失敗：{exc}") from exc

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
                "report_options": [
                    "個資蒐集",
                    "詐騙",
                    "騷擾",
                    "可疑電話",
                    "銀行信貸騷擾",
                    "企業假冒",
                    "其他",
                ],
            }

        first_reported_at = (
            result["first_reported_at"].strftime("%Y-%m-%d %H:%M:%S")
            if result["first_reported_at"] is not None
            else None
        )
        last_reported_at = (
            result["last_reported_at"].strftime("%Y-%m-%d %H:%M:%S")
            if result["last_reported_at"] is not None
            else None
        )

        return {
            "phone_number": result["phone_number"],
            "status": result["status"],
            "phone_type": result["phone_type"],
            "total_reports": result["total_reports"],
            "first_reported_at": first_reported_at,
            "last_reported_at": last_reported_at,
            "owner_name": result["owner_name"],
            "can_report": False,
            "report_options": [
                "個資蒐集",
                "詐騙",
                "騷擾",
                "可疑電話",
                "銀行信貸騷擾",
                "企業假冒",
                "其他",
            ],
        }

    def report_suspicious(self, phone_number: str, phone_type: str, other_type: str | None = None) -> dict:
        if phone_type == "其他":
            phone_type = other_type or phone_type

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with engine.begin() as conn:
                phone_row = conn.execute(
                    text(
                        "SELECT id, status FROM phone WHERE phone_number = :phone_number"
                    ),
                    {"phone_number": phone_number},
                ).mappings().first()

                if phone_row:
                    phone_id = phone_row["id"]
                    conn.execute(
                        text(
                            "UPDATE phone SET phone_type = :phone_type, status = 'black' "
                            "WHERE id = :phone_id"
                        ),
                        {"phone_type": phone_type, "phone_id": phone_id},
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO phone(phone_number, phone_type, status) "
                            "VALUES (:phone_number, :phone_type, 'black')"
                        ),
                        {"phone_number": phone_number, "phone_type": phone_type},
                    )
                    phone_id = conn.execute(
                        text(
                            "SELECT id FROM phone WHERE phone_number = :phone_number"
                        ),
                        {"phone_number": phone_number},
                    ).scalar()

                blacklist_row = conn.execute(
                    text(
                        "SELECT total_reports FROM blacklist WHERE phone_id = :phone_id"
                    ),
                    {"phone_id": phone_id},
                ).mappings().first()

                if blacklist_row:
                    conn.execute(
                        text(
                            "UPDATE blacklist "
                            "SET total_reports = total_reports + 1, last_reported_at = :last_reported_at "
                            "WHERE phone_id = :phone_id"
                        ),
                        {"last_reported_at": now, "phone_id": phone_id},
                    )
                    total_reports = blacklist_row["total_reports"] + 1
                    first_reported_at = conn.execute(
                        text(
                            "SELECT first_reported_at FROM blacklist WHERE phone_id = :phone_id"
                        ),
                        {"phone_id": phone_id},
                    ).scalar()
                else:
                    conn.execute(
                        text(
                            "INSERT INTO blacklist(phone_id, total_reports, first_reported_at, last_reported_at) "
                            "VALUES (:phone_id, 1, :first_reported_at, :last_reported_at)"
                        ),
                        {
                            "phone_id": phone_id,
                            "first_reported_at": now,
                            "last_reported_at": now,
                        },
                    )
                    total_reports = 1
                    first_reported_at = now
        except OperationalError as exc:
            raise RuntimeError(f"資料庫連線失敗：{exc}") from exc

        return {
            "phone_number": phone_number,
            "status": "black",
            "total_reports": total_reports,
            "report_time": now,
            "message": "可疑電話號碼已回報，已新增或更新黑名單記錄。",
        }


phone_service = PhoneService()
