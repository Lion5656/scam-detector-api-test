import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.config import settings


class CaseRepository:
    def __init__(self, path: str | None = None):
        self._path = Path(path or settings.CASE_LOG_PATH)

    def append_case(self, payload: dict) -> str:
        case_id = str(uuid4())
        envelope = {
            "case_id": case_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")

        return case_id


case_repository = CaseRepository()
